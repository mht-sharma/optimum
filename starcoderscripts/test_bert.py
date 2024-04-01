from functools import partial

from transformers import AutoTokenizer, WhisperProcessor

from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig, AutoCalibrationConfig
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy


# Quantization args
full_fit = True
calibration_num_shards = 1
max_length = 512
save_dir = "quant_bert"

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
processor = AutoTokenizer.from_pretrained(model_id)

onnx_model = "onnx_bert"
tokenizer = AutoTokenizer.from_pretrained(model_id)
quantizer = ORTQuantizer.from_pretrained(onnx_model, file_name="model.onnx")
qconfig = AutoQuantizationConfig.arm64(is_static=True, per_channel=False, nodes_to_quantize=["Matmul", "Add"])


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

from pdb import set_trace

set_trace()

calibration_dataset = quantizer.get_calibration_dataset(
    "glue",
    dataset_config_name="sst2",
    preprocess_function=partial(preprocess_fn, tokenizer=tokenizer, max_length=max_length),
    num_samples=50,
    dataset_split="train",
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
            onnx_augmented_model_name=args.output.joinpath("model_augmented.onnx").as_posix(),
            batch_size=args.calibration_batch_size,
            use_external_data_format=args.external_data_format,
            operators_to_quantize=qconfig.operators_to_quantize,
        )
    ranges = quantizer.compute_ranges()

model_quantized_path = quantizer.quantize(
    save_dir=save_dir,
    calibration_tensors_range=ranges,
    quantization_config=qconfig,
)
