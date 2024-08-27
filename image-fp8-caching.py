import gc
import os
import uuid

import torch
from diffusers import FluxTransformer2DModel, FluxPipeline
from optimum.quanto import freeze, qfloat8, quantize
from transformers import T5EncoderModel, QuantoConfig

bfl_repo = "black-forest-labs/FLUX.1-dev"
dtype = torch.float16

# fixme: loading quantized models doesn't work yet
def quantize_and_save(model, model_name):
    if not os.path.exists(model_name):
        quantize(model, qfloat8)
        freeze(model)
        path = f"models/{model_name}"
        quant_config = QuantoConfig(weights="float8", activations="float8")
        model.save_pretrained(path, max_shard_size="15GB", quant_config=quant_config)
        print(f"Model saved to {model_name}")


get_model_path = lambda model_name: f"models/{model_name}"

# =================================================================================
transformer = FluxTransformer2DModel.from_single_file(
    "https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-dev-fp8.safetensors", torch_dtype=dtype,
    token=os.getenv("HF_TOKEN"))
transformer_name = "flux1-dev-fp8"
transformer_path = get_model_path(transformer_name)
quantize_and_save(transformer, transformer_path)

print(f"Loading transformer from {transformer_path}...")
transformer = FluxTransformer2DModel.from_pretrained(transformer_path, torch_dtype=dtype)
print("Done.")

# =================================================================================
text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_1", torch_dtype=dtype)
clip_name = "t5-clip-fp8"
clip_path = get_model_path(clip_name)
quantize_and_save(text_encoder_2, clip_path)

print(f"Loading text_encoder_2 from {clip_path}...")
text_encoder_2 = T5EncoderModel.from_pretrained(clip_path, torch_dtype=dtype)
print("Done.")

# =================================================================================
pipe = FluxPipeline.from_pretrained(bfl_repo, transformer=None, text_encoder_2=None, torch_dtype=dtype)
pipe.transformer = transformer
pipe.text_encoder_2 = text_encoder_2

pipe.enable_model_cpu_offload()


def cleanup():
    torch.cuda.empty_cache()
    gc.collect()


def generate_images(prompt: str, width: int = 1024, height: int = 1024, num_inference_steps: int = 30,
                    num_varaitions: int = 1):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    images = pipe(
        prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_varaitions,
        guidance_scale=3.5,
        output_type="pil",
        generator=torch.Generator(device=device).manual_seed(0)
    ).images

    # image path should be "<uuid>/flux-dev-<image_index>.png"
    rd_uuid = uuid.uuid4()
    for image_index, image in enumerate(images):
        image.save(f"{rd_uuid}/flux-dev-{image_index}.png")

    cleanup()


def main():
    while True:
        prompt = input("Enter a prompt: ")
        if prompt == "exit":
            break
        elif prompt == "example":
            prompt = "A cat holding a sign saying hello world"

        generate_images(prompt)


main()
