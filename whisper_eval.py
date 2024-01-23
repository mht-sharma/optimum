import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import onnxruntime as onnxrt
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoProcessor, GenerationConfig, WhisperForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


model_name = "openai/whisper-tiny.en"
config = AutoConfig.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

batch_size = 1
encoder_num_attention_heads = 6
decoder_num_attention_heads = 6
hidden_size = 384
encoder_sequence_length = 1500
decoder_max_length = 448
num_hidden_layers = 4

encoder_shape = (
    batch_size,
    encoder_num_attention_heads,
    encoder_sequence_length,
    hidden_size // encoder_num_attention_heads,
)
decoder_shape = (
    batch_size,
    decoder_num_attention_heads,
    decoder_max_length,
    hidden_size // decoder_num_attention_heads,
)


# load dataset
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
idx = 4
inputs = processor.feature_extractor(ds[idx]["audio"]["array"], return_tensors="pt")
input_features = inputs.input_features


onnx_model_path = "/home/ubuntu/optimum/output_whisper_smooth_quant_24_oct_static/"
# onnx_model_path = "/home/ubuntu/whisper-static-shapes-onnx"
encoder_model_path = Path(onnx_model_path) / "encoder_model.onnx"
decoder_model_path = Path(onnx_model_path) / "decoder_model_quantized_static_scatter.onnx"

print(decoder_model_path)


class ORTEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main_input_name = "input_features"
        self.session = onnxrt.InferenceSession(encoder_model_path, providers=["CPUExecutionProvider"])
        self.output_names = {output_key.name: idx for idx, output_key in enumerate(self.session.get_outputs())}

    def forward(
        self,
        input_features: torch.FloatTensor,
        **kwargs,
    ) -> BaseModelOutput:
        onnx_inputs = {"input_features": input_features.cpu().detach().numpy()}

        # Run inference
        outputs = self.session.run(None, onnx_inputs)
        last_hidden_state = torch.from_numpy(outputs[self.output_names["last_hidden_state"]])

        return BaseModelOutput(last_hidden_state=last_hidden_state)


class ORTDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.session = onnxrt.InferenceSession(decoder_model_path, providers=["CPUExecutionProvider"])

        self.generation_config = GenerationConfig.from_model_config(config)
        self.max_length = decoder_max_length

        self.input_names = {input_key.name: idx for idx, input_key in enumerate(self.session.get_inputs())}
        self.output_names = {output_key.name: idx for idx, output_key in enumerate(self.session.get_outputs())}
        self.key_value_input_names = [key for key in self.input_names if (".key" in key) or (".value" in key)]
        self.key_value_output_names = [key for key in self.output_names if (".key" in key) or (".value" in key)]

        self.reset()

    def reset(self):
        # Set the start model inputs
        self.decoder_attention_mask = np.zeros((batch_size, self.max_length)).astype(np.int64)
        self.decoder_attention_mask[0, 0] = 1
        self.position_ids = np.array([[0]]).astype(np.int64)

        # Set the input / output names
        self.num_pkv = 4

    def prepare_pkv(self):
        decoder_key_value = torch.rand(*decoder_shape).to(torch.float32)
        encoder_key_value = torch.rand(*encoder_shape).to(torch.float32)

        past_key_values = []
        repeat_count = len(self.key_value_input_names) // 4
        past_key_values = tuple(
            (decoder_key_value, decoder_key_value, encoder_key_value, encoder_key_value) for _ in range(repeat_count)
        )

        return tuple(past_key_values)

    def forward(
        self,
        input_ids: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    ) -> Seq2SeqLMOutput:
        if past_key_values is None:
            self.reset()

        if self.position_ids[0][0] == self.max_length:
            logits = torch.zeros((len(input_ids), 1, config.vocab_size))
            logits[:, :, config.eos_token_id] = 1

            return Seq2SeqLMOutput(logits=logits, past_key_values=past_key_values)

        onnx_inputs = {"decoder_input_ids": input_ids.cpu().detach().numpy()}

        onnx_inputs["position_ids"] = self.position_ids
        onnx_inputs["decoder_attention_mask"] = self.decoder_attention_mask
        onnx_inputs["encoder_outputs"] = encoder_hidden_states.cpu().detach().numpy()

        if self.position_ids[0][0] == 0:
            past_key_values = self.prepare_pkv()

        past_key_values = tuple(
            past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer
        )

        for input_name, past_key_value in zip(self.key_value_input_names, past_key_values):
            onnx_inputs[input_name] = past_key_value.cpu().detach().numpy()

        # Run inference
        outputs = self.session.run(None, onnx_inputs)

        logits = torch.from_numpy(outputs[self.output_names["logits"]])

        out_past_key_values = tuple(
            torch.from_numpy(outputs[self.output_names[key]]) for key in self.key_value_output_names
        )

        if self.position_ids[0][0] == 0:
            out_past_key_values = tuple(
                out_past_key_values[i : i + self.num_pkv] for i in range(0, len(out_past_key_values), self.num_pkv)
            )
        else:
            out_past_key_values = tuple(
                out_past_key_values[i : i + 2] + past_key_values[i + 2 : i + 4]
                for i in range(0, len(out_past_key_values), self.num_pkv)
            )

        if self.position_ids[0][0] < self.max_length - 1:
            self.decoder_attention_mask[:, self.position_ids[0][0] + 1] = 1
        self.position_ids += 1

        return Seq2SeqLMOutput(logits=logits, past_key_values=out_past_key_values)


class ORTModelForWhisper(WhisperForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        config = AutoConfig.from_pretrained(model_name)
        super().__init__(config)

        self.encoder = ORTEncoder()
        self.decoder = ORTDecoder()

    def get_encoder(self):
        return self.encoder

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_features=input_features)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids[:, -1:],
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            past_key_values=past_key_values,
        )

        return Seq2SeqLMOutput(
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
        )

    def can_generate(self):
        return True

    def reset(self):
        self.decoder.reset()


model_ort = ORTModelForWhisper()
model = WhisperForConditionalGeneration.from_pretrained(model_name)


def test_ort():
    model = ORTModelForWhisper()

    generated_ids = model.generate(input_features)
    model_output = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("ORT: ", model_output, generated_ids)


def test_original():
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    generated_ids = model.generate(input_features)
    model_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("Torch: ", model_output, generated_ids)


def evaluate(model_test):
    import evaluate

    wer_metric = evaluate.load("wer")

    test_dataset = load_dataset(
        "librispeech_asr",
        name="clean",
        split="test",
    )

    # num_samples = min(2, len(test_dataset))
    # test_dataset = test_dataset.shuffle(seed=10).select(range(num_samples))

    predictions = []
    references = []
    print(len(test_dataset))
    for i, data in enumerate(test_dataset):
        audio_samples = data["audio"]

        arrays = audio_samples["array"]
        sampling_rate = audio_samples["sampling_rate"]

        input_feat = processor(arrays, sampling_rate=sampling_rate, return_tensors="pt").input_features

        generated_ids = model_test.generate(input_feat)
        model_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        model_output = processor.tokenizer._normalize(model_output)
        reference_output = processor.tokenizer._normalize(data["text"])
        predictions.append(model_output)
        references.append(reference_output)

        print(i)

        # print("Reference: ", reference_output)
        # print("Output: ", model_output)

    wer = wer_metric.compute(references=references, predictions=predictions)
    wer = round(100 * wer, 2)

    print("WER:", wer, "%")


# evaluate(model_ort)


# test_original()
test_ort()


print(encoder_model_path)

print(decoder_model_path)
