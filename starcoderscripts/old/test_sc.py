import torch


model_name = "static_quantization_whisper"
model = torch.jit.load("static_quantization_whisper/pytorch_model.bin")
print(model)
# model = WhisperForConditionalGeneration.from_pretrained(model_name)
