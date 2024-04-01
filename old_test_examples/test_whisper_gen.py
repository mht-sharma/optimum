from datasets import load_dataset
from transformers import AutoFeatureExtractor, AutoTokenizer, WhisperForConditionalGeneration, WhisperProcessor
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy

from optimum.onnxruntime import ORTModelForSpeechSeq2Seq


model_id = "openai/whisper-tiny.en"

# load model and processor
processor = WhisperProcessor.from_pretrained(model_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id)

# load dummy dataset and read audio files
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]
input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features
max_length = 64
text_inputs = "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."
tokens = tokenizer(
    text_inputs,
    padding=PaddingStrategy.MAX_LENGTH,
    truncation=TruncationStrategy.LONGEST_FIRST,
    max_length=max_length,
)
print(tokens)


def generate(model):
    # generate token ids
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
    print(transcription)


onnx_model_path = "onnx_whisper"
onnx_model_path = "custom_whisper_onnx"
model_ort = ORTModelForSpeechSeq2Seq.from_pretrained(
    onnx_model_path,
    encoder_file_name="encoder_model_static.onnx",
    decoder_file_name="decoder_model.onnx",
    use_cache=False,
)


# onnx_model_path = "output_whisper_smooth_quant_3_oct"
# model_ort_quant = ORTModelForSpeechSeq2Seq.from_pretrained(
#     onnx_model_path,
#     encoder_file_name="encoder_model_static_quantized.onnx",
#     decoder_file_name="decoder_model_static_quantized.onnx",
#     use_cache=False,
# )
generate(model)
generate(model_ort)
# generate(model_ort_quant)


# model_ort.save_pretrained("test_pls_work2")
# # model_ort.push_to_hub("test_pls_work2", repository_id="mohitsha/whisper-tiny-smooth-quant")

# pipe = transformers.pipeline(
#     "automatic-speech-recognition", model=model, feature_extractor=feature_extractor, tokenizer=tokenizer
# )
# from optimum.pipelines import pipeline as ort_pipeline


# pipe_ort = ort_pipeline(
#     "automatic-speech-recognition", model=model_ort, feature_extractor=feature_extractor, tokenizer=tokenizer
# )
# print(pipe(sample["array"]))
# print(pipe_ort(sample["array"]))

# from datasets import load_dataset
# from evaluate import evaluator


# def eval(pipe_test):
#     task_evaluator = evaluator("automatic-speech-recognition")
#     # data = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="validation[:40]", download_mode=FORCE_REDOWNLOAD)
#     results = task_evaluator.compute(
#         model_or_pipeline=pipe_test,
#         data=ds,
#         input_column="audio",
#         label_column="text",
#         metric="wer",
#     )
#     print(results)


# # eval(pipe)
# # eval(pipe_ort)
