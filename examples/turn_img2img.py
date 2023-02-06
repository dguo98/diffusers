import torch
import os
import requests
from PIL import Image
from io import BytesIO
from glob import glob
from tqdm import tqdm
from IPython import embed

from diffusers import StableDiffusionImg2ImgPipeline

device="cuda"
model = input("input model path:") or "runwayml/stable-diffusion-v1-5"

# HACK(demi)
model = "nitrosocke/Arcane-Diffusion"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model).to(device)

image_folder = input("input image folder:") or "/nlp/scr/demiguo/explore/sd-3d/demi-turn"
print("image_folder=", image_folder)
#zuck-same-video-300/frame_00028.jpg"


output_path = input("output folder path:") or "/nlp/scr/demiguo/explore/sd-3d/demi-turn-output"
os.makedirs(output_path, exist_ok=True)


image_paths = sorted(glob(f"{image_folder}/*.jpg"))
total = len(image_paths)

for i, image_path in tqdm(enumerate(image_paths), desc="img2img all images"):
    image_id = image_path.split("/")[-1].replace(".png", "")
    try:
        init_image = Image.open(image_path).convert("RGB")
    except:
        print("unable to open", image_path, " i=", i)
        embed()
        continue

    # center crop
    width, height = init_image.size
    new_size = min(width, height) // 2
    left = width // 2 - new_size
    right = width // 2 + new_size
    top = height // 2 - new_size
    bottom = height // 2 + new_size
    
    init_image = init_image.crop((left, top, right,  bottom))   
    init_image = init_image.resize((768, 768))
    if i == 0:
        init_image.save(f"{output_path}/init_image_preprocess.png")
    
    if os.path.exists(f"{output_path}/{image_id}_0.jpg"):
        print("pass", image_id)
        continue

    # TODO(demi): check if we need to resize
    prompt = "arcane style"
    strength = 0.5
    gs = 20. / total * i + 10
    gs = 25
    try:
        images = pipe(prompt=prompt, init_image=init_image, strength=strength, guidance_scale=gs, num_images_per_prompt=1).images

        #print("image_path=", image_path, "image_id=", image_id)
        for i,image in enumerate(images):
            image.save(f"{output_path}/{image_id}_{i}.jpg")
    except:
        print("failed at ", image_id)

