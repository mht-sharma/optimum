from functools import partial

from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy

from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoCalibrationConfig, AutoQuantizationConfig


# Quantization args
full_fit = True
calibration_num_shards = 1
max_length = 512
save_dir = "quant_bert_qdq_sc"

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
processor = AutoTokenizer.from_pretrained(model_id)

onnx_model = "onnx_bert"
tokenizer = AutoTokenizer.from_pretrained(model_id)
quantizer = ORTQuantizer.from_pretrained(onnx_model, file_name="model.onnx")
qconfig = AutoQuantizationConfig.arm64(is_static=True, per_channel=False, nodes_to_quantize=["MatMul"])


def preprocess_fn(ex, tokenizer, max_length):
    out = tokenizer(
        ex["sentence"],
        padding=PaddingStrategy.MAX_LENGTH,
        truncation=TruncationStrategy.LONGEST_FIRST,
        max_length=max_length,
    )
    return out


calibration_dataset = quantizer.get_calibration_dataset(
    "glue",
    dataset_config_name="sst2",
    preprocess_function=partial(preprocess_fn, tokenizer=tokenizer, max_length=max_length),
    num_samples=50,
    dataset_split="train",
)

# apply smooth quant
quantizer.apply_smooth_quant(
    dataset=calibration_dataset,
    save_dir=save_dir,
    quantization_config=qconfig,
    # batch_size=2
)

# from pdb import set_trace; set_trace()

calibration_dataset = quantizer.get_calibration_dataset(
    "glue",
    dataset_config_name="sst2",
    preprocess_function=partial(preprocess_fn, tokenizer=tokenizer, max_length=max_length),
    num_samples=50,
    dataset_split="train",
)

calibration_config = AutoCalibrationConfig.minmax(calibration_dataset)

ranges = quantizer.fit(
    dataset=calibration_dataset,
    calibration_config=calibration_config,
    operators_to_quantize=qconfig.operators_to_quantize,
)

model_quantized_path = quantizer.quantize(
    save_dir=save_dir,
    calibration_tensors_range=ranges,
    quantization_config=qconfig,
)
