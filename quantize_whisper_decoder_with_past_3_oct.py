from functools import partial
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tqdm import trange
from transformers import AutoConfig, AutoTokenizer, WhisperProcessor

from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoCalibrationConfig, AutoQuantizationConfig


# Quantization settings
full_fit = True
calibration_num_shards = 4
save_dir = "output_whisper_smooth_quant_24_oct_staticxsax"
ops_to_quantize_smooth_quant = ["MatMul"]
ops_to_quantize = ["MatMul"]
model_id = "openai/whisper-tiny.en"

# Initialize WhisperProcessor and ORTQuantizer
processor = WhisperProcessor.from_pretrained(model_id)
config = AutoConfig.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

onnx_model_path = "/home/ubuntu/whisper-static-shapes-onnx"
onnx_model_path_dynamic = "onnx_whisper"

tokenizer = AutoTokenizer.from_pretrained(model_id)
quantizer = ORTQuantizer.from_pretrained(onnx_model_path, file_name="decoder_model.onnx")

# Configure AutoQuantization
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=True, per_channel=False, operators_to_quantize=ops_to_quantize)
qconfig.smooth_quant_op_types = ops_to_quantize_smooth_quant
qconfig.nodes_to_exclude = ["/proj_out/MatMul"]


def preprocess_function(examples, processor, tokenizer, max_length=448, position_id=64):
    audio_samples = examples["audio"]

    arrays = audio_samples["array"]
    sampling_rate = audio_samples["sampling_rate"]

    inputs = processor(arrays, sampling_rate=sampling_rate)

    onnx_inputs_encoder = {}
    onnx_inputs_encoder["input_features"] = inputs["input_features"]

    session = ort.InferenceSession(Path(onnx_model_path_dynamic) / "encoder_model.onnx")
    sess_output_encoder = session.run(None, onnx_inputs_encoder)

    # Tokenize the text input
    text_inputs = examples["text"]
    input_ids = tokenizer(text_inputs).input_ids

    if len(input_ids) < max_length:
        idx = len(input_ids) // 2 + 1
    else:
        idx = position_id

    nidx = idx
    import random

    if random.random() < 0.10:
        idx = 0
        nidx = 1

    onnx_inputs_decoder = {}
    onnx_inputs_decoder["input_ids"] = [input_ids[:nidx]]
    onnx_inputs_decoder["encoder_hidden_states"] = sess_output_encoder[0]

    session = ort.InferenceSession(Path(onnx_model_path_dynamic) / "decoder_model.onnx")
    sess_output_decoder = session.run(None, onnx_inputs_decoder)

    output = {}

    output["decoder_input_ids"] = [input_ids[idx]]
    output["decoder_attention_mask"] = np.zeros((max_length)).astype(np.int64)
    output["decoder_attention_mask"][: idx + 1] = 1
    output["encoder_outputs"] = sess_output_encoder[0][0]

    for i in range(int((len(sess_output_decoder) - 1) / 4)):

        def modify_tensor_shape(tensor):
            if tensor.shape[1] < max_length:
                random_values = np.zeros((tensor.shape[0], max_length - tensor.shape[1], tensor.shape[2]))
                return np.concatenate((tensor, random_values), axis=1)
            return tensor

        # Modify both decoder.key and decoder.value
        output["past_key_values.{}.decoder.key".format(i)] = modify_tensor_shape(sess_output_decoder[i * 4 + 1][0])
        output["past_key_values.{}.decoder.value".format(i)] = modify_tensor_shape(sess_output_decoder[i * 4 + 2][0])
        output["past_key_values.{}.encoder.key".format(i)] = sess_output_decoder[i * 4 + 3][0]
        output["past_key_values.{}.encoder.value".format(i)] = sess_output_decoder[i * 4 + 4][0]

    output["position_ids"] = np.array([0]).astype(np.int64)
    output["position_ids"][0] = idx

    return output


# Generate a calibration dataset
calibration_dataset = quantizer.get_calibration_dataset(
    "librispeech_asr",
    dataset_config_name="clean",
    preprocess_function=partial(preprocess_function, processor=processor, tokenizer=tokenizer),
    num_samples=128,
    dataset_split="train.100",
    preprocess_batch=False,
)

# # Apply smooth quantization to the model
# quantizer.apply_smooth_quant(
#     dataset=calibration_dataset,
#     save_dir=save_dir,
#     quantization_config=qconfig,
# )

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
