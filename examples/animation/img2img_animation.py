import argparse
from datetime import datetime
import sys
import os
from typing import List

import cv2
import av
import numpy as np
import torch
import tensorflow as tf
from PIL import Image

from diffusers import StableDiffusionImg2ImgPipeline

sys.path.append("frame-interpolation")
from eval import interpolator as film_interpolator, util as film_util

MODEL_CACHE = "diffusers-cache"

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='runwayml/stable-diffusion-v1-5', help="Model id or path")
# Start with either an image or a prompt
parser.add_argument('--init_image', type=str, default=None, help="Image path to start the animation with")
parser.add_argument('--prompts', type=str, nargs='+', help="Prompts for the animation")
parser.add_argument('--width', type=int, default=512, help="Width of output image")
parser.add_argument('--height', type=int, default=512, help="Height of output image")
parser.add_argument('--num_inference_steps', type=int, default=50, help="Number of denoising steps (1~500)")
parser.add_argument(
    '--strength', type=float, default=0.8,
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
parser.add_argument('--output_path', type=str, help="Directory for output files")


class Img2ImgAnimation:
    def __init__(
        self, model, output_path,
        # from https://drive.google.com/drive/folders/1i9Go1YI2qiFWeT5QtywNFmYAA74bhXWj?usp=sharing
        film_path="saved_model",
        **kwargs
    ):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model,
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

        # Create output directory
        self.output_path = output_path
        os.makedirs(os.path.join(output_path, "tmp"), exist_ok=True)

    @torch.no_grad()
    def predict(
        self,
        init_image: str,
        prompts: List[str],
        width: int,
        height: int,
        num_inference_steps: int,
        strength: float,
        num_animation_frames: int,
        num_interpolation_steps: int,
        guidance_scale: float,
        gif_frames_per_second: int,
        gif_ping_pong: bool,
        film_interpolation: bool,
        intermediate_output: bool,
        seed: int,
        output_format: str,
        **kwargs
    ) -> str:
        """Run multiple predictions on the model to form animation"""
        # 1. Check inputs. Raise error if not correct
        if not isinstance(prompts, list):
            raise ValueError(f"`prompts` has to be of type `list` but is {type(prompts)}")

        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        # 2. Define call parameters
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        batch_size = 1
        num_images_per_prompt = 1
        device = torch.device("cuda")
        generator = torch.Generator(device).manual_seed(seed)
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = [self.pipe._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
        ) for prompt in prompts]

        # 4. Preprocess image
        image = load_img(init_image, (width, height))

        # 5. set timesteps
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_denoise_steps = self.pipe.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        num_warmup_steps = len(timesteps) - num_denoise_steps * self.pipe.scheduler.order

        # 6. Prepare latent variables
        init_latents = self.pipe.prepare_latents(
            image, latent_timestep, batch_size, num_images_per_prompt,
            prompt_embeds[0].dtype, device, generator
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta=0.)

        # 8. Denoising loop
        frames_latents = []
        for prompt_idx in range(len(prompts) - 1):
            for frame_idx in range(num_animation_frames):
                print(f"Generating frame {frame_idx + 1} of prompt {prompt_idx}")
                cur_prompt_embed = slerp(
                    (frame_idx + 1) / num_animation_frames,
                    prompt_embeds[prompt_idx],
                    prompt_embeds[prompt_idx + 1],
                )
                latents = init_latents
                with self.pipe.progress_bar(total=num_denoise_steps) as progress_bar:
                    for i, t in enumerate(timesteps):
                        # expand the latents if we are doing classifier free guidance
                        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                        latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)

                        # predict the noise residual
                        noise_pred = self.pipe.unet(latent_model_input, t, encoder_hidden_states=cur_prompt_embed).sample

                        # perform guidance
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = self.pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                        # call the callback, if provided
                        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.pipe.scheduler.order == 0):
                            progress_bar.update()
                print("latents.shape=", latents.shape)
                frames_latents.append(latents)

                # optionally save frames in separate files
                if intermediate_output:
                    image = self.pipe.decode_latents(latents)
                    save_pil_image(
                        self.pipe.numpy_to_pil(image)[0],
                        path=os.path.join(
                            self.output_path, "tmp",
                            f"prompt_{prompt_idx}-frame_{frame_idx}.png"
                        )
                    )

                # reset timesteps
                self.pipe.scheduler.set_timesteps(num_inference_steps, device=device)

        # 9. Post-processing
        if gif_ping_pong:
            frames_latents.extend(frames_latents[-2::-1])

        # Decode images and interpolate between animation frames
        if film_interpolation:
            images = self.interpolate_film(frames_latents, num_interpolation_steps)
        else:
            images = self.interpolate_latents(
                frames_latents, num_interpolation_steps
            )

        # 10. Run safety checker
        # image, has_nsfw_concept = self.pipe.run_safety_checker(images[0], device, prompt_embeds[0].dtype)

        # Save the video
        if output_format == "gif":
            return self.save_gif(images, gif_frames_per_second)
        else:
            return self.save_mp4(images, gif_frames_per_second, width, height)

    def save_mp4(self, images, fps, width, height):
        print("Saving MP4")
        output_path = os.path.join(self.output_path, f"output-{get_datetime()}.mp4")

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
            self.pipe.numpy_to_pil(img)[0] for img in images
        ]

        output_path = os.path.join(self.output_path, f"output-{get_datetime()}.gif")
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
                image = self.pipe.decode_latents(latents)[0]
                images.append(image)
        return images

    def interpolate_film(self, frames_latents, num_interpolation_steps):
        print("Interpolating images with FILM")
        images = [
            self.pipe.decode_latents(lat)[0].astype("float32")
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
    if resize is not None:
        w, h = resize
    w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 8
    print(f"resizing to ({w}, {h})")
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


def save_pil_image(image, path):
    image.save(path)


def get_datetime():
    now = datetime.now()
    return now.strftime("%Y-%m-%d-%H:%M:%S")


if __name__ == "__main__":
    args = parser.parse_args()
    args = vars(args)
    predictor = Img2ImgAnimation(**args)
    print(predictor.predict(**args))
