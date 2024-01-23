from functools import partial
from pathlib import Path

from tqdm import trange
from transformers import AutoTokenizer, WhisperProcessor

from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoCalibrationConfig, AutoQuantizationConfig


# Quantization settings
full_fit = True
calibration_num_shards = 4
save_dir = "output_whisper_smooth_quant_4_oct_static_sl_448_512"
ops_to_quantize = ["MatMul"]
model_id = "openai/whisper-tiny.en"

# Initialize WhisperProcessor and ORTQuantizer
processor = WhisperProcessor.from_pretrained(model_id)
onnx_model_path = "/home/ubuntu/whisper-static-shapes-onnx/"
tokenizer = AutoTokenizer.from_pretrained(model_id)
quantizer = ORTQuantizer.from_pretrained(onnx_model_path, file_name="encoder_model.onnx")

# Configure AutoQuantization
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=True, per_channel=False, operators_to_quantize=ops_to_quantize)
qconfig.smooth_quant_op_types = ops_to_quantize


# Preprocess audio data for calibration
def preprocess_function(examples, processor):
    audio_samples = examples["audio"]

    arrays = [sample["array"] for sample in audio_samples]
    sampling_rate = audio_samples[0]["sampling_rate"]

    output = processor(arrays, sampling_rate=sampling_rate, return_tensors="pt")
    return output


# Generate a calibration dataset
calibration_dataset = quantizer.get_calibration_dataset(
    "librispeech_asr",
    dataset_config_name="clean",
    preprocess_function=partial(preprocess_function, processor=processor),
    num_samples=512,
    dataset_split="train.100",
)

# Apply smooth quantization to the model
quantizer.apply_smooth_quant(
    dataset=calibration_dataset,
    save_dir=save_dir,
    quantization_config=qconfig,
)

# Configure calibration
calibration_config = AutoCalibrationConfig.minmax(calibration_dataset)

# Perform full fit or partial fit based on the 'full_fit' flag
if full_fit:
    ranges = quantizer.fit(
        dataset=calibration_dataset,
        calibration_config=calibration_config,
        operators_to_quantize=qconfig.operators_to_quantize,
    )
else:
    for i in trange(calibration_num_shards):
        shard = calibration_dataset.shard(calibration_num_shards, i, keep_in_memory=True)
        quantizer.partial_fit(
            dataset=shard,
            calibration_config=calibration_config,
            onnx_augmented_model_name=Path(save_dir).joinpath("model_augmented.onnx").as_posix(),
            batch_size=4,
            use_external_data_format=False,
            operators_to_quantize=qconfig.operators_to_quantize,
        )
    ranges = quantizer.compute_ranges()

# Quantize the model
quantized_model_path = quantizer.quantize(
    save_dir=save_dir,
    calibration_tensors_range=ranges,
    quantization_config=qconfig,
)
