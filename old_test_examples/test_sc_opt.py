from transformers import AutoTokenizer, pipeline


generator = pipeline("text-generation", model="facebook/opt-125m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
text = "What are we having for dinner? Is it salmon or fire"
print(generator(text))

onnx_model_path = "output_opt_smooth_quant"


from optimum.onnxruntime import ORTModelForCausalLM


model_ort = ORTModelForCausalLM.from_pretrained(
    onnx_model_path,
    decoder_file_name="decoder_model_quantized.onnx",
    use_cache=False,
    use_io_binding=False,
)

generator2 = pipeline("text-generation", model=model_ort, tokenizer=tokenizer)
print(generator2(text))
model_ort.save_pretrained("opt_onnx")

model_ort.push_to_hub("opt_onnx", repository_id="mohitsha/opt-125m-smooth-quant")
