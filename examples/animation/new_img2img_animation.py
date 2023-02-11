import argparse
import sys
import os
from typing import Optional, List, Iterator

import cv2
import av
import numpy as np
import torch
from torch import autocast
import tensorflow as tf
from diffusers import PNDMScheduler, LMSDiscreteScheduler
from PIL import Image

from diffusers import StableDiffusionImg2ImgTextAnimatePipeline

sys.path.append("frame-interpolation")
from eval import interpolator as film_interpolator, util as film_util

MODEL_CACHE = "diffusers-cache"

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='runwayml/stable-diffusion-v1-5', help="Model id or path")
# Start with either an image or a prompt
parser.add_argument('--init_image', type=str, default=None, help="Image path to start the animation with")
parser.add_argument('--prompt_start', type=str, default=None, help="Prompt to start the animation with")
parser.add_argument(
    '--prompt_end', type=str, default=None,
    help="Prompt to end the animation with. You can include multiple prompts by separating the prompts with | (the 'pipe' character)")
parser.add_argument('--width', type=int, default=512, help="Width of output image")
parser.add_argument('--height', type=int, default=512, help="Height of output image")
parser.add_argument('--num_inference_steps', type=int, default=50, help="Number of denoising steps (1~500)")
parser.add_argument(
    '--prompt_strength', type=float, default=0.8,
    help="Lower prompt strength generates more coherent gifs, higher respects prompts more but can be jumpy")
parser.add_argument('--num_animation_frames', type=int, default=10, help="Number of frames to animate (2~50)")
parser.add_argument(
    '--num_interpolation_steps', type=int, default=5,
    help="Number of steps to interpolate between animation frames (1~50)")
parser.add_argument('--guidance_scale', type=float, default=7.5, help="Scale for classifier-free guidance (1~20)")
parser.add_argument('--gif_frames_per_second', type=float, default=20, help="Frames/second in output GIF (1~50)")
parser.add_argument(
    '--gif_ping_pong', action='store_true',
    help="Whether to reverse the animation and go back to the beginning before looping")
parser.add_argument(
    '--film_interpolation', action='store_true',
    help="Whether to use FILM for between-frame interpolation (film-net.github.io)")
parser.add_argument(
    '--intermediate_output', action='store_true',
    help="Whether to display intermediate outputs during generation")
parser.add_argument('--seed', type=int, default=None, help="Random seed. Leave blank to randomize the seed")
parser.add_argument('--output_format', type=str, default="mp4", help="Output file format (gif or mp4)")


class Predictor:
    def __init__(
        self, model_id_or_path="runwayml/stable-diffusion-v1-5",
        # from https://drive.google.com/drive/folders/1i9Go1YI2qiFWeT5QtywNFmYAA74bhXWj?usp=sharing
        film_path="saved_model"
    ):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        self.pipe = StableDiffusionImg2ImgTextAnimatePipeline.from_pretrained(
            model_id_or_path,
            torch_dtype=torch.float16,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to("cuda")

        # Stop tensorflow eagerly taking all GPU memory
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        print("Loading interpolator...")
        self.interpolator = film_interpolator.Interpolator(
            film_path,
            None,
        )

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        init_image: str,
        prompt_start: str,
        prompt_end: str,
        width: int,
        height: int,
        num_inference_steps: int,
        prompt_strength: float,
        num_animation_frames: int,
        num_interpolation_steps: int,
        guidance_scale: float,
        gif_frames_per_second: int,
        gif_ping_pong: bool,
        film_interpolation: bool,
        intermediate_output: bool,
        seed: int,
        output_format: str
    ) -> str:
        """Run a single prediction on the model"""
        with torch.autocast("cuda"), torch.inference_mode():
            if seed is None:
                seed = int.from_bytes(os.urandom(2), "big")
            print(f"Using seed: {seed}")
            generator = torch.Generator("cuda").manual_seed(seed)

            batch_size = 1

            # Generate initial latents to start to generate animation frames from

            do_classifier_free_guidance = guidance_scale > 1.0

            prompts = [prompt_start] + [
                p.strip() for p in prompt_end.strip().split("|")
            ]


            # NB(demi): dummy hyper-params
            device="cuda"
            num_images_per_prompt=1
            negative_prompt=None

            keyframe_text_embeddings = []
            for prompt in prompts:
                print("prompt=", prompt)
                keyframe_text_embeddings.append(
                    self.pipe._encode_prompt(
                        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
                    )
                )

            pil_init_image = Image.open(init_image).convert("RGB").resize((width, height))

            final_image, _, init_latents, init_final_latents = self.pipe(prompts[0], pil_init_image, 
                prompt_strength, num_inference_steps, guidance_scale, 
                generator=generator, get_latents=True)

            print("init_latents.shape=", init_latents.shape)

            final_image[0].save(f"tmp/init_final_image.jpg")  # HACK(demi): save final init image, should be img2img
 
            if init_image is None:
                print("no init image")
                # TODO(demi): initialize with random, get init_latents and init_final_latents
                raise NotImplementedError

            # Generate animation frames
            frames_latents = [init_final_latents]

            if intermediate_output:
                for i, latents in enumerate(frames_latents):
                    image = self.pipe.latents_to_image(latents)
                    save_pil_image(
                        self.pipe.numpy_to_pil(image)[0],
                        path=f"tmp/output-init-{i}.png",
                    )

            for keyframe in range(len(prompts) - 1):
                for i in range(num_animation_frames):
                    print(f"Generating frame {i + 1} of keyframe {keyframe}")
                    text_embeddings = slerp(
                        (i + 1) / num_animation_frames,
                        keyframe_text_embeddings[keyframe],
                        keyframe_text_embeddings[keyframe + 1],
                    )
                    
                    print("new prompt=",prompts[keyframe])
                    _, _, final_latents = self.pipe(
                        prompt=prompts[keyframe],  # HACK(demi)
                        init_image=pil_init_image,
                        strength=prompt_strength,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        #prompt_embeds=text_embeddings,
                        #reuse_latents=init_latents HACK(demi)
                        )

                    # de-noise this frame
                    frames_latents.append(latents)
                    if intermediate_output:
                        image = self.pipe.latents_to_image(latents)
                        save_pil_image(
                            self.pipe.numpy_to_pil(image)[0],
                            path=f"tmp/output-{keyframe}-{i + 1}.png",
                        )

            if gif_ping_pong:
                frames_latents.extend(frames_latents[-2::-1])

            # Decode images by interpolate between animation frames
            if film_interpolation:
                images = self.interpolate_film(frames_latents, num_interpolation_steps)
            else:
                images = self.interpolate_latents(
                    frames_latents, num_interpolation_steps
                )

            # Save the video
            if output_format == "gif":
                return self.save_gif(images, gif_frames_per_second)
            else:
                return self.save_mp4(images, gif_frames_per_second, width, height)

    def save_mp4(self, images, fps, width, height):
        print("Saving MP4")
        output_path = "tmp/output.mp4"

        output = av.open(output_path, "w")
        stream = output.add_stream(
            "h264", rate=fps, options={"crf": "17", "tune": "film"}
        )
        # stream.bit_rate = 8000000
        # stream.bit_rate = 16000000
        stream.width = width
        stream.height = height

        for i, image in enumerate(images):
            image = (image * 255).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            packet = stream.encode(frame)
            output.mux(packet)

        # flush
        packet = stream.encode(None)
        output.mux(packet)
        output.close()

        return output_path

    def save_gif(self, images, fps):
        print("Saving GIF")
        pil_images = [
            self.pipe.numpy_to_pil(img.astype("float32"))[0] for img in images
        ]

        output_path = "tmp/video.gif"
        gif_frame_duration = int(1000 / fps)

        with open(output_path, "wb") as f:
            pil_images[0].save(
                fp=f,
                format="GIF",
                append_images=pil_images[1:],
                save_all=True,
                duration=gif_frame_duration,
                loop=0,
            )

        return output_path

    def interpolate_latents(self, frames_latents, num_interpolation_steps):
        print("Interpolating images from latents")
        images = []
        for i in range(len(frames_latents) - 1):
            latents_start = frames_latents[i]
            latents_end = frames_latents[i + 1]
            for j in range(num_interpolation_steps):
                x = j / num_interpolation_steps
                latents = latents_start * (1 - x) + latents_end * x
                image = self.pipe.latents_to_image(latents)[0]
                images.append(image)
        return images

    def interpolate_film(self, frames_latents, num_interpolation_steps):
        print("Interpolating images with FILM")
        images = [
            self.pipe.latents_to_image(lat)[0].astype("float32")
            for lat in frames_latents
        ]
        if num_interpolation_steps == 0:
            return images

        num_recursion_steps = max(int(np.ceil(np.log2(num_interpolation_steps))), 1)
        images = film_util.interpolate_recursively_from_memory(
            images, num_recursion_steps, self.interpolator
        )
        images = [img.clip(0, 1) for img in images]
        return images


def make_scheduler(num_inference_steps, from_scheduler=None):
    scheduler = PNDMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
    )
    scheduler.set_timesteps(num_inference_steps)
    if from_scheduler:
        scheduler.cur_model_output = from_scheduler.cur_model_output
        scheduler.counter = from_scheduler.counter
        scheduler.cur_sample = from_scheduler.cur_sample
        scheduler.ets = from_scheduler.ets[:]
    return scheduler


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """helper function to spherically interpolate two arrays v1 v2"""
    # from https://gist.github.com/nateraw/c989468b74c616ebbc6474aa8cdd9e53

    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2


def load_img(path, resize=None):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    if resize is None:
        w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    else:
        w, h = resize
    print(f"resizing to ({w}, {h})")
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


def save_pil_image(image, path):
    image.save(path)


if __name__ == "__main__":
    args = parser.parse_args()
    args = vars(args)
    predictor = Predictor(args['model'])
    args.pop('model')
    print(predictor.predict(**args))
