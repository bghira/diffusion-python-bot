# classes/image_generator.py
import numpy as np
import textwrap
from torchvision.transforms.functional import pad

import torch
import time
from diffusers import StableDiffusionPipeline, StableDiffusionImageVariationPipeline
import requests
from huggingface_hub import hf_hub_url, HfApi, snapshot_download
from .app_config import AppConfig
from PIL import Image
import logging
import os
from io import BytesIO
from .model_downloader import ModelDownloader

# Set up the logging configuration
logging.basicConfig(level=logging.INFO)

class ImageGenerator:
    def __init__(self, device="cuda", torch_dtype=torch.float16):
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype

    def get_variation_pipe(self, model_id):
        logging.info("Clearing the CUDA cache...")
        torch.cuda.empty_cache()
        logging.info("Generating a new variation pipe...")
        pipe = StableDiffusionImageVariationPipeline.from_pretrained(model_id, torch_dtype=self.torch_dtype)
        logging.info("Moving the pipe to the device...")
        pipe.to(self.device)
        logging.info("Return the pipe...")
        return pipe

    def generate_image_variations(self, prompt, input_image, guidance_scale=3, num_inference_steps=25):
        logging.info("Initializing image variation generation pipeline...")
        pipe = self.get_variation_pipe("lambdalabs/sd-image-variations-diffusers")
        input_image = pad(input_image, (input_image.size[0] // 2, input_image.size[1] // 2))
        # Generate image variations
        generated_images = pipe(prompt=prompt, image=input_image, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images

        return generated_images

    def get_pipe(self, model_id):
        logging.info("Clearing the CUDA cache...")
        torch.cuda.empty_cache()
        logging.info("Generating a new pipe...")
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=self.torch_dtype)
        logging.info("Moving the pipe to the device...")
        pipe.to(self.device)
        pipe.safety_checker = lambda images, clip_input: (images, False)
        logging.info("Return the pipe...")
        return pipe
    def generate_image(self, prompt, model_id=None, resolution=None, negative_prompt=None, steps=50, positive_prompt=None, max_retries=1, retry_delay=10):
        logging.info("Initializing image generation pipeline...")
        pipe = self.get_pipe(model_id)
        logging.info("Copied pipe to the local context")
        if resolution is not None:
            side_x = resolution['width']
            if side_x > 1280:
                side_x = 1280
            side_y = resolution['height']
            if side_y > 720:
                side_y = 720
        else:
            side_x = side_y = None
        logging.info("Set custom resolution")
        # Combine the main prompt and positive_prompt if provided
        entire_prompt = prompt
        if positive_prompt is not None:
            entire_prompt = str(prompt) + ' , ' + str(positive_prompt)
        for attempt in range(1, max_retries + 1):
            try:
                logging.info(f"Attempt {attempt}: Generating image...")
                # Use the processed_prompt instead of the raw prompt
                image = pipe(prompt=entire_prompt, height=side_y, width=side_x, num_inference_steps=int(float(steps)), negative_prompt=negative_prompt).images[0]
                del pipe
                torch.cuda.empty_cache()

                logging.info("Image generation successful!")
                return image
            except Exception as e:
                logging.error(f"Attempt {attempt}: Image generation failed with error: {e}")
                del pipe
                torch.cuda.empty_cache()
                if attempt < max_retries:
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError("Maximum retries reached, image generation failed")

    def get_available_models(self):
        config = AppConfig()
        base_dir = config.get_local_model_path()

        model_prefix = "models--"
        local_models = []
        for subdir in os.listdir(base_dir):
            if subdir.startswith(model_prefix):
                subdir_path = os.path.join(base_dir, subdir)
                if os.path.isdir(subdir_path):
                    # Remove "models-" prefix and replace "--" with "/"
                    model_id = subdir[len(model_prefix):].replace("--", "/")
                    local_models.append(model_id)

        return local_models