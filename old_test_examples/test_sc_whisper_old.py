from functools import partial
from pathlib import Path

from tqdm import trange
from transformers import AutoTokenizer, WhisperProcessor

from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoCalibrationConfig, AutoQuantizationConfig


# Quantization args
full_fit = True
calibration_num_shards = 4
save_dir = "output_whisper_smooth_quant"
ops_to_quantize = ["MatMul"]

model_id = "openai/whisper-tiny.en"
processor = WhisperProcessor.from_pretrained(model_id)

onnx_model = "onnx_whisper"
tokenizer = AutoTokenizer.from_pretrained(model_id)
quantizer = ORTQuantizer.from_pretrained(onnx_model, file_name="encoder_model.onnx")
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=True, per_channel=False, operators_to_quantize=ops_to_quantize)
qconfig.smooth_quant_op_types = ops_to_quantize


def preprocess_function(examples, processor):
    sample = examples["audio"]

    array = []
    sampling_rate = []
    for sam in sample:
        array.append(sam["array"])
        sampling_rate.append(sam["sampling_rate"])
    out = processor(array, sampling_rate=sampling_rate[0], return_tensors="pt")
    # out = processor(sample["array"], sampling_rate=sample["sampling_rate"])
    return out


calibration_dataset = quantizer.get_calibration_dataset(
    "librispeech_asr",
    dataset_config_name="clean",
    preprocess_function=partial(preprocess_function, processor=processor),
    num_samples=50,
    dataset_split="train.100",
    # preprocess_batch=False,
)

# apply smooth quant
quantizer.apply_smooth_quant(
    dataset=calibration_dataset,
    save_dir=save_dir,
    quantization_config=qconfig,
    # batch_size=2
)

calibration_config = AutoCalibrationConfig.minmax(calibration_dataset)

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

model_quantized_path = quantizer.quantize(
    save_dir=save_dir,
    calibration_tensors_range=ranges,
    quantization_config=qconfig,
)
