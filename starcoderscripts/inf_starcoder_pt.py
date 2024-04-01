from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


# checkpoint = "hf-internal-testing/tiny-random-GPTBigCodeModel"
# checkpoint = "bigcode/gpt_bigcode-santacoder"
checkpoint = "bigcode/starcoderbase-1b"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)
config = AutoConfig.from_pretrained(checkpoint)
tokenizer.pad_token ="[PAD]"
tokenizer.padding_side = "left"
text = "Write hello world code in c++"
inputs = tokenizer(text, padding="max_length", max_length=128, return_tensors="pt")

outputs = model.generate(**inputs, use_cache=True, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
