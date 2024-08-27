import gc
import os
import uuid
import time

import torch
from diffusers import FluxTransformer2DModel, FluxPipeline
from optimum.quanto import freeze, qfloat8, quantize
from transformers import T5EncoderModel, QuantoConfig

bfl_repo = "black-forest-labs/FLUX.1-dev"
dtype = torch.float16

# =================================================================================
# measure startup time
start_time = time.time()
print("Starting up...")

transformer = FluxTransformer2DModel.from_single_file(
    "https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-dev-fp8.safetensors", torch_dtype=dtype,
    token=os.getenv("HF_TOKEN"))
quantize(transformer, qfloat8)
freeze(transformer)

text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype)
quantize(text_encoder_2, qfloat8)
freeze(text_encoder_2)

pipe = FluxPipeline.from_pretrained(bfl_repo, transformer=None, text_encoder_2=None, torch_dtype=dtype)
pipe.transformer = transformer
pipe.text_encoder_2 = text_encoder_2

pipe.enable_model_cpu_offload()

print(f"Startup time: {time.time() - start_time} seconds")


# =================================================================================


def cleanup():
    torch.cuda.empty_cache()
    gc.collect()


def generate_images(prompt: str, width: int = 1024, height: int = 1024, num_inference_steps: int = 30,
                    num_varaitions: int = 2):
    print("Generating images...")
    current_time = time.time()

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

    # Erstelle das Verzeichnis, bevor Bilder gespeichert werden
    rd_uuid = uuid.uuid4()
    directory = f"images/{rd_uuid}"
    os.makedirs(directory, exist_ok=True)

    for image_index, image in enumerate(images):
        try:
            image.save(f"{directory}/flux-dev-{image_index}.png")
        except Exception as e:
            print(f"Fehler beim Speichern von Bild {image_index}: {e}")

    print(f"Generated {len(images)} images in {time.time() - current_time} seconds")
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
