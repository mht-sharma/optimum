from pathlib import Path
from typing import Dict

from transformers import AutoConfig

from optimum.exporters.onnx import main_export
from optimum.exporters.onnx.base import ConfigBehavior
from optimum.exporters.onnx.model_configs import WhisperOnnxConfig


class CustomWhisperOnnxConfig(WhisperOnnxConfig):
    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = super().inputs

        if self._behavior is not ConfigBehavior.ENCODER:
            if self.use_past_in_inputs:
                common_inputs["decoder_attention_mask"] = {0: "batch_size"}
            else:
                common_inputs["decoder_attention_mask"] = {0: "batch_size", 1: "decoder_sequence_length"}

        return common_inputs


model_id = "openai/whisper-tiny.en"
config = AutoConfig.from_pretrained(model_id)

custom_whisper_onnx_config = CustomWhisperOnnxConfig(
    config=config,
    task="automatic-speech-recognition",
)

encoder_config = custom_whisper_onnx_config.with_behavior("encoder")
decoder_config = custom_whisper_onnx_config.with_behavior("decoder", use_past=False)

custom_onnx_configs = {
    "encoder_model": encoder_config,
    "decoder_model": decoder_config,
}

main_export(
    model_id,
    output="custom_whisper_onnx",
    task="automatic-speech-recognition",
    no_post_process=True,
    custom_onnx_configs=custom_onnx_configs,
)


def reshape(model_path, input_shape_dict, output_shape_dict):
    import onnx
    from onnx import shape_inference
    from onnx.tools import update_model_dims

    model_path = Path(model_path)
    static_model_path = model_path.parent / (model_path.stem + "_static" + model_path.suffix)

    model = onnx.load(model_path)

    updated_model = update_model_dims.update_inputs_outputs_dims(model, input_shape_dict, output_shape_dict)
    shape_inferered_model = shape_inference.infer_shapes(updated_model)

    onnx.save(shape_inferered_model, static_model_path)


seq_lenth = 64
batch_size = 1
reshape(
    "custom_whisper_onnx/encoder_model.onnx",
    input_shape_dict={"input_features": [batch_size, 80, 3000]},
    output_shape_dict={"last_hidden_state": [batch_size, 1500, 384]},
)
reshape(
    "custom_whisper_onnx/decoder_model.onnx",
    input_shape_dict={
        "input_ids": [batch_size, seq_lenth],
        "decoder_attention_mask": [batch_size, seq_lenth],
        "encoder_hidden_states": [batch_size, 1500, 384],
    },
    output_shape_dict={"logits": [batch_size, seq_lenth, 51864]},
)
