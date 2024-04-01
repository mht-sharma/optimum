from functools import partial

import torch
from neural_compressor.config import PostTrainingQuantConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from optimum.intel import INCModelForSequenceClassification, INCQuantizer


model_name = "sshleifer/tiny-distilbert-base-cased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
with torch.no_grad():
    logits1 = model(**inputs).logits

predicted_class_id1 = logits1.argmax().item()
print(predicted_class_id1, logits1)
# The directory where the quantized model will be saved
save_dir = "static_quantization1"


def preprocess_function(examples, tokenizer):
    return tokenizer(examples["sentence"], padding="max_length", max_length=128, truncation=True)


# Load the quantization configuration detailing the quantization we wish to apply
# quantization_config = PostTrainingQuantConfig(approach="static")
recipes = {"smooth_quant": True, "smooth_quant_args": {"alpha": 0.5, "folding": True}}
quantization_config = PostTrainingQuantConfig(approach="static", backend="ipex", recipes=recipes)
quantizer = INCQuantizer.from_pretrained(model)
# Generate the calibration dataset needed for the calibration step
calibration_dataset = quantizer.get_calibration_dataset(
    "glue",
    dataset_config_name="sst2",
    preprocess_function=partial(preprocess_function, tokenizer=tokenizer),
    num_samples=8,
    dataset_split="train",
)
quantizer = INCQuantizer.from_pretrained(model)
# Apply static quantization and save the resulting model
quantizer.quantize(
    quantization_config=quantization_config,
    calibration_dataset=calibration_dataset,
    save_directory=save_dir,
)

model = INCModelForSequenceClassification.from_pretrained(save_dir)
with torch.no_grad():
    logits = model(**inputs)

print(logits)
print(predicted_class_id1, logits1)
