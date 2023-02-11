import torch
import requests
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline

device="cuda"
model = input("input model path:") or "runwayml/stable-diffusion-v1-5"
MODEL_CACHE="animation/diffusers-cache"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model,
    torch_dtype=torch.float16,
    cache_dir=MODEL_CACHE,
    local_files_only=True).to(device)

seed = int(input("seed:")) or 0

image_path = input("input image path:") or "/nlp/scr/demiguo/explore/animation/init.jpg"

output_path = input("output folder path:") or "/nlp/scr/demiguo/explore/animation/tmp"

ids=0

init_image = Image.open(image_path).convert("RGB")
init_image = init_image.resize((1024, 576))
generator = torch.Generator("cuda").manual_seed(seed)

"""
width, height = init_image.size
new_size = min(width, height) // 2
left = width // 2 - new_size
right = width // 2 + new_size
top = height // 2 - new_size
bottom = height // 2 + new_size

init_image = init_image.crop((left, top, right,  bottom))   
init_image = init_image.resize((768, 768))
"""
while True:
    prompt = input("input prompt:") or "arcane style"
    strength = float(input("strength (default 0.75):") or 0.2)
    gs = float(input("guidance scale (default 7.5):") or 7)

    images = pipe(prompt=prompt, init_image=init_image, strength=strength, guidance_scale=gs, num_images_per_prompt=1, generator=generator,
        num_inference_steps=50).images
    for i in range(1):
        ids+=1
        images[i].save(f"{output_path}/img2img_{ids}.png") 
