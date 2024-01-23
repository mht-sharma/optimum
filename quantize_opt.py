from functools import partial
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tqdm import trange
from transformers import AutoConfig, AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy

from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoCalibrationConfig, AutoQuantizationConfig


# Quantization settings
full_fit = True
calibration_num_shards = 4
save_dir = "output_opt_smooth_quant"
ops_to_quantize = ["MatMul", "Add"]
model_id = "facebook/opt-125m"
max_length = 2048

config = AutoConfig.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

onnx_model_path = "/home/ubuntu/opt-125m-onnx-static-shapes/"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "left"  # NOTE: this is strictly required!
quantizer = ORTQuantizer.from_pretrained(onnx_model_path, file_name="decoder_with_past_model.onnx")

# Configure AutoQuantization
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=True, per_channel=False, operators_to_quantize=ops_to_quantize)
qconfig.smooth_quant_op_types = ops_to_quantize


# Preprocess audio data for calibration
def preprocess_function(examples, tokenizer, max_length=128):
    from pdb import set_trace

    set_trace()
    text_inputs = examples["text"]
    inputs = tokenizer(
        text_inputs,
        padding=PaddingStrategy.MAX_LENGTH,
        truncation=TruncationStrategy.LONGEST_FIRST,
        max_length=max_length,
    )
    onnx_inputs_decoder = {}
    onnx_inputs_decoder["input_ids"] = inputs["input_ids"]
    onnx_inputs_decoder["attention_mask"] = inputs["attention_mask"]

    position_ids = (inputs["attention_mask"].cumsum(-1) * inputs["attention_mask"]) - 1
    onnx_inputs_decoder["position_ids"] = position_ids + 2

    session = ort.InferenceSession(Path(onnx_model_path) / "decoder_model.onnx")
    sess_output_decoder = session.run(None, onnx_inputs_decoder)

    np.argmax(sess_output_decoder[0], axis=-1)[:, -1].item()

    output = {}

    output["input_ids"] = [input_ids[idx]]
    output["decoder_attention_mask"] = np.zeros((max_length)).astype(np.int64)
    output["decoder_attention_mask"][: idx + 1] = 1
    output["encoder_hidden_states"] = sess_output_encoder[0][0]

    for i in range(int((len(sess_output_decoder) - 1) / 2)):

        def modify_tensor_shape(tensor):
            if tensor.shape[1] < max_length:
                random_values = np.zeros((tensor.shape[0], max_length - tensor.shape[1], tensor.shape[2]))
                return np.concatenate((tensor, random_values), axis=1)
            return tensor

        # Modify both decoder.key and decoder.value
        output["past_key_values.{}.key".format(i)] = modify_tensor_shape(sess_output_decoder[i * 2 + 1][0])
        output["past_key_values.{}.value".format(i)] = modify_tensor_shape(sess_output_decoder[i * 2 + 2][0])

    return output


# Generate a calibration dataset
calibration_dataset = quantizer.get_calibration_dataset(
    "allenai/c4",
    dataset_config_name="en",
    preprocess_function=partial(preprocess_function, tokenizer=tokenizer, max_length=max_length),
    num_samples=10,
    # data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
    dataset_split="train",
    preprocess_batch=False,
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
