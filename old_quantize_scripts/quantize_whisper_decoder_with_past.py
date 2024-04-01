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
save_dir = "output_whisper_smooth_quant_23"
ops_to_quantize = ["MatMul"]
model_id = "openai/whisper-tiny.en"

# Initialize WhisperProcessor and ORTQuantizer
processor = WhisperProcessor.from_pretrained(model_id)
config = AutoConfig.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

onnx_model_path = "onnx_whisper"
tokenizer = AutoTokenizer.from_pretrained(model_id)
quantizer = ORTQuantizer.from_pretrained(onnx_model_path, file_name="decoder_with_past_model.onnx")

# Configure AutoQuantization
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=True, per_channel=False, operators_to_quantize=ops_to_quantize)
qconfig.smooth_quant_op_types = ops_to_quantize


def preprocess_function_padded(examples, processor, tokenizer, max_length=64):
    audio_samples = examples["audio"]

    arrays = [sample["array"] for sample in audio_samples]
    sampling_rate = audio_samples[0]["sampling_rate"]

    inputs = processor(arrays, sampling_rate=sampling_rate)
    onnx_inputs = {}
    onnx_inputs["input_features"] = inputs["input_features"]

    session = ort.InferenceSession(Path(onnx_model_path) / "encoder_model.onnx")
    sess_output_encoder = session.run(None, onnx_inputs)

    text_inputs = examples["text"]
    input_ids = tokenizer(
        text_inputs,
        padding=PaddingStrategy.MAX_LENGTH,
        truncation=TruncationStrategy.LONGEST_FIRST,
        max_length=max_length,
        return_tensors="np",
    ).input_ids

    onnx_inputs_decoder = {}

    onnx_inputs_decoder["input_ids"] = input_ids[:, :-1]
    onnx_inputs_decoder["encoder_hidden_states"] = sess_output_encoder[0]

    session = ort.InferenceSession(Path(onnx_model_path) / "decoder_model.onnx")
    sess_output_decoder = session.run(None, onnx_inputs_decoder)

    output = {}

    output["input_ids"] = input_ids[:, -1].reshape((len(input_ids), 1))
    for i in range(int((len(sess_output_decoder) - 1) / 4)):
        output["past_key_values.{}.decoder.key".format(i)] = sess_output_decoder[i * 4 + 1]
        output["past_key_values.{}.decoder.value".format(i)] = sess_output_decoder[i * 4 + 2]
        output["past_key_values.{}.encoder.key".format(i)] = sess_output_decoder[i * 4 + 3]
        output["past_key_values.{}.encoder.value".format(i)] = sess_output_decoder[i * 4 + 4]

    # print(input_ids[:10])

    return output


def preprocess_function(examples, processor, tokenizer):
    audio_samples = examples["audio"]

    arrays = audio_samples["array"]
    sampling_rate = audio_samples["sampling_rate"]

    inputs = processor(arrays, sampling_rate=sampling_rate)

    # Tokenize the text input
    text_inputs = examples["text"]
    input_ids = tokenizer(text_inputs).input_ids

    onnx_inputs_encoder = {}
    onnx_inputs_encoder["input_features"] = inputs["input_features"]

    session = ort.InferenceSession(Path(onnx_model_path) / "encoder_model.onnx")
    sess_output_encoder = session.run(None, onnx_inputs_encoder)

    onnx_inputs_decoder = {}
    import random

    random_length = random.randint(1, len(input_ids) - 2)
    # random_length = -1

    onnx_inputs_decoder["input_ids"] = [input_ids[:random_length]]
    onnx_inputs_decoder["encoder_hidden_states"] = sess_output_encoder[0]

    session = ort.InferenceSession(Path(onnx_model_path) / "decoder_model.onnx")
    sess_output_decoder = session.run(None, onnx_inputs_decoder)

    output = {}

    output["input_ids"] = [input_ids[random_length]]
    output["encoder_hidden_states"] = sess_output_encoder[0][0]

    print(output["input_ids"])
    for i in range(int((len(sess_output_decoder) - 1) / 4)):
        output["past_key_values.{}.decoder.key".format(i)] = sess_output_decoder[i * 4 + 1][0]
        output["past_key_values.{}.decoder.value".format(i)] = sess_output_decoder[i * 4 + 2][0]
        output["past_key_values.{}.encoder.key".format(i)] = sess_output_decoder[i * 4 + 3][0]
        output["past_key_values.{}.encoder.value".format(i)] = sess_output_decoder[i * 4 + 4][0]

    return output


# Generate a calibration dataset
calibration_dataset = quantizer.get_calibration_dataset(
    "librispeech_asr",
    dataset_config_name="clean",
    preprocess_function=partial(preprocess_function, processor=processor, tokenizer=tokenizer),
    num_samples=2,
    dataset_split="train.100",
    preprocess_batch=False,
)

# Apply smooth quantization to the model
quantizer.apply_smooth_quant(
    dataset=calibration_dataset,
    save_dir=save_dir,
    quantization_config=qconfig,
)

# Configure calibration
calibration_config = AutoCalibrationConfig.minmax(calibration_dataset, moving_average=False)
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
