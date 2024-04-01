import transformers
from datasets import load_dataset
from transformers import AutoFeatureExtractor, AutoTokenizer, WhisperForConditionalGeneration, WhisperProcessor

from optimum.onnxruntime import ORTModelForSpeechSeq2Seq


# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny.en")
tokenizer = AutoTokenizer.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

# load dummy dataset and read audio files
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]
input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features

# generate token ids
predicted_ids = model.generate(input_features)
# decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)

print(transcription)

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

onnx_model_path = "output_whisper_smooth_quant_23"
model_ort = ORTModelForSpeechSeq2Seq.from_pretrained(
    onnx_model_path,
    encoder_file_name="encoder_model_quantized.onnx",
    decoder_file_name="decoder_model_quantized.onnx",
    decoder_with_past_file_name="decoder_with_past_model_quantized.onnx",
    use_cache=False,
)
# model_ort = ORTModelForSpeechSeq2Seq.from_pretrained(
#     "openai/whisper-tiny.en",
#     export=True,
#     use_cache=False
# )

# generate token ids
predicted_ids = model_ort.generate(input_features)
# decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
print(predicted_ids)
print(transcription)

# model_ort.save_pretrained("test_pls_work2")


# model_ort.push_to_hub("test_pls_work2", repository_id="mohitsha/whisper-tiny-smooth-quant")
pipe = transformers.pipeline(
    "automatic-speech-recognition", model=model, feature_extractor=feature_extractor, tokenizer=tokenizer
)
from optimum.pipelines import pipeline as ort_pipeline


pipe_ort = ort_pipeline(
    "automatic-speech-recognition", model=model_ort, feature_extractor=feature_extractor, tokenizer=tokenizer
)
print(pipe(sample["array"]))
print(pipe_ort(sample["array"]))
# print(pipe_ort(sample["array"]))
from datasets import load_dataset
from evaluate import evaluator


# from datasets.DownloadMode import FORCE_REDOWNLOAD


def eval(pipe_test):
    task_evaluator = evaluator("automatic-speech-recognition")
    # data = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="validation[:40]", download_mode=FORCE_REDOWNLOAD)
    results = task_evaluator.compute(
        model_or_pipeline=pipe_test,
        data=ds,
        input_column="audio",
        label_column="text",
        metric="wer",
    )
    print(results)


eval(pipe)
eval(pipe_ort)
