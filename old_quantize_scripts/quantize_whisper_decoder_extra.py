from functools import partial
from pathlib import Path

import onnxruntime as ort
from tqdm import trange
from transformers import AutoConfig, AutoTokenizer, WhisperProcessor
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy

from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoCalibrationConfig, AutoQuantizationConfig


# Quantization settings
full_fit = True
calibration_num_shards = 4
save_dir = "output_whisper_smooth_quant"
ops_to_quantize = ["MatMul"]
model_id = "openai/whisper-tiny.en"
max_length = 256

# Initialize WhisperProcessor and ORTQuantizer
processor = WhisperProcessor.from_pretrained(model_id)
config = AutoConfig.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

onnx_model_path = "onnx_whisper"
tokenizer = AutoTokenizer.from_pretrained(model_id)
quantizer = ORTQuantizer.from_pretrained(onnx_model_path, file_name="decoder_model.onnx")

# Configure AutoQuantization
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=True, per_channel=False, operators_to_quantize=ops_to_quantize)
qconfig.smooth_quant_op_types = ops_to_quantize


# Preprocess audio data for calibration
def preprocess_function(examples, processor, tokenizer, max_length):
    audio_samples = examples["audio"]
    examples["text"]

    arrays = [sample["array"] for sample in audio_samples]
    sampling_rate = audio_samples[0]["sampling_rate"]

    inputs = processor(arrays, sampling_rate=sampling_rate)
    onnx_inputs = {}
    onnx_inputs["input_features"] = inputs["input_features"]

    session = ort.InferenceSession(Path(onnx_model_path) / "encoder_model.onnx")
    sess_output = session.run(None, onnx_inputs)

    output = {}
    output["input_ids"] = [[50257] for i in range(len(sess_output[0]))]
    output["encoder_hidden_states"] = sess_output[0]
    return output


# Preprocess audio data for calibration
def preprocess_function_padded(examples, processor, tokenizer, max_length):
    audio_samples = examples["audio"]
    examples["text"]

    arrays = [sample["array"] for sample in audio_samples]
    sampling_rate = audio_samples[0]["sampling_rate"]

    inputs = processor(arrays, sampling_rate=sampling_rate)
    onnx_inputs = {}
    onnx_inputs["input_features"] = inputs["input_features"]

    session = ort.InferenceSession(Path(onnx_model_path) / "encoder_model.onnx")
    sess_output = session.run(None, onnx_inputs)

    text_inputs = examples["text"]
    input_ids = tokenizer(
        text_inputs,
        padding=PaddingStrategy.MAX_LENGTH,
        truncation=TruncationStrategy.LONGEST_FIRST,
        max_length=max_length,
    ).input_ids

    output = {}
    output["input_ids"] = input_ids
    output["encoder_hidden_states"] = sess_output[0]
    return output


# Generate a calibration dataset
calibration_dataset = quantizer.get_calibration_dataset(
    "librispeech_asr",
    dataset_config_name="clean",
    preprocess_function=partial(preprocess_function, processor=processor, tokenizer=tokenizer, max_length=max_length),
    num_samples=50,
    dataset_split="train.100",
    preprocess_batch=True,
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
