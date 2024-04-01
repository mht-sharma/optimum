from functools import partial

import torch
from datasets import load_dataset
from neural_compressor.config import PostTrainingQuantConfig
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, WhisperForConditionalGeneration

from optimum.intel import INCModelForSpeechSeq2Seq, INCQuantizer


model_name = "openai/whisper-tiny.en"
processor = AutoProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
model.config.use_cache = False
# model = model.model.encoder
# from pdb import set_trace; set_trace()

# The directory where the quantized model will be saved
save_dir = "static_quantization_whisper"


# load dummy dataset and read audio files
ds = load_dataset("librispeech_asr", "clean", split="validation")
sample = ds[10]["audio"]
input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features

_, sequence_length = input_features[0].shape
am = [torch.tensor(sequence_length) for _ in range(1)]
attention_mask = torch.tensor(am)
decoder_input_ids = torch.tensor([[torch.tensor(50257)]])

example_inputs = {
    "input_features": input_features,
    "attention_mask": attention_mask,
    "decoder_input_ids": decoder_input_ids,
}


def calibration_func(model):
    calib_num = 5
    with torch.no_grad():
        for i in range(calib_num):
            model(**example_inputs)


def preprocess_function(examples, processor, tokenizer):
    # from pdb import set_trace; set_trace()
    sample = examples["audio"]
    array = []
    sampling_rate = []
    decoder_input_ids = []
    for sam in sample:
        array.append(sam["array"])
        sampling_rate.append(sam["sampling_rate"])
        decoder_input_ids.append([torch.tensor(50257)])
    out = processor(array, sampling_rate=sampling_rate[0], return_tensors="pt")
    _, sequence_length = out["input_features"][0].shape
    am = [torch.tensor(sequence_length) for _ in range(len(sample))]
    out["attention_mask"] = am
    # out["attention_mask"] = [None for _ in range(len(sample))]
    out["decoder_input_ids"] = torch.tensor(decoder_input_ids)
    return out


op_type_dict = {"Embedding": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}}}
op_type_dict["add"] = {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}}
op_type_dict["Linear"] = {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}}
op_type_dict["matmul"] = {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}}

# Load the quantization configuration detailing the quantization we wish to apply
# quantization_config = PostTrainingQuantConfig(approach="static")
recipes = {"smooth_quant": True, "smooth_quant_args": {"alpha": 0.5, "folding": True}}
quantization_config = PostTrainingQuantConfig(
    approach="static", backend="ipex", recipes=recipes, op_type_dict=op_type_dict
)
quantizer = INCQuantizer.from_pretrained(model, calibration_fn=calibration_func)

# Generate the calibration dataset needed for the calibration step
calibration_dataset = quantizer.get_calibration_dataset(
    "librispeech_asr",
    dataset_config_name="clean",
    preprocess_function=partial(preprocess_function, processor=processor, tokenizer=tokenizer),
    num_samples=100,
    dataset_split="validation",
)
# Apply static quantization and save the resulting model
quantizer.quantize(
    quantization_config=quantization_config,
    calibration_dataset=calibration_dataset,
    save_directory=save_dir,
)

model_inc = INCModelForSpeechSeq2Seq.from_pretrained(save_dir)

# from pdb import set_trace; set_trace()

with torch.no_grad():
    # generate token ids
    out1 = model(input_features=input_features, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
    print(out1)

with torch.no_grad():
    out = model_inc(input_features=input_features, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)

    print(out)


def calculate_errors(tensor1, tensor2):
    max_diff = torch.amax(torch.abs(tensor1 - tensor2))
    return max_diff


max_diff = calculate_errors(out1["logits"], out["logits"])
print("max_diff Error:", max_diff)
max_diff = calculate_errors(out1["encoder_last_hidden_state"], out["encoder_last_hidden_state"])
print("max_diff Error:", max_diff)
