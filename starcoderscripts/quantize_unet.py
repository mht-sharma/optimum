from optimum.onnxruntime import ORTQuantizer, ORTStableDiffusionPipeline
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from pathlib import Path
from transformers import PretrainedConfig

save_path = Path("sd_onnx")
model_id = "runwayml/stable-diffusion-v1-5"
pipeline = ORTStableDiffusionPipeline.from_pretrained(model_id, export=True)
pipeline.save_pretrained(save_path)

quantizer = ORTQuantizer(save_path / "unet/model.onnx", config=PretrainedConfig.from_dict(pipeline.unet.config))

dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

model_quantized_path = quantizer.quantize(
    save_dir=save_path / "unet",
    quantization_config=dqconfig,
)