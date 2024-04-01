from functools import partial

from transformers import AutoTokenizer, WhisperProcessor

from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig, AutoCalibrationConfig


# Quantization args
full_fit = True
calibration_num_shards = 1
save_dir = "quant_whisper"

model_id = "openai/whisper-tiny.en"
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")

onnx_model = "onnx_whisper"
tokenizer = AutoTokenizer.from_pretrained(model_id)
quantizer = ORTQuantizer.from_pretrained(onnx_model, file_name="encoder_model.onnx")
qconfig = AutoQuantizationConfig.arm64(is_static=True, per_channel=False, nodes_to_quantize=["Matmul"])


def preprocess_function(examples, processor):
    sample = examples["audio"]

    # from pdb import set_trace

    # set_trace()
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
    num_samples=8,
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
