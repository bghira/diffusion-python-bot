# classes/image_generator.py

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
        
    def get_pipe(self, model_id):
        torch.cuda.empty_cache()
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=self.torch_dtype)
        pipe.to(self.device)
        pipe.safety_checker = lambda images, clip_input: (images, False)
        return pipe
    def get_variation_pipe(self):
        torch.cuda.empty_cache()
        pipe = StableDiffusionImageVariationPipeline.from_pretrained(model_id, torch_dtype=self.torch_dtype)
        pipe.to(self.device)
        pipe.safety_checker = lambda images, clip_input: (images, False)
        return pipe
    def generate_image(self, prompt, model_id=None, resolution=None, negative_prompt=None, steps=50, positive_prompt=None, max_retries=1, retry_delay=10):
        pipe = self.get_pipe(model_id)

        if resolution is not None:
            side_x = resolution['width']
            if side_x > 1280:
                side_x = 1280
            side_y = resolution['height']
            if side_y > 720:
                side_y = 720
        else:
            side_x = side_y = None

        for attempt in range(1, max_retries + 1):
            try:
                # Generate the image via the model.
                entire_prompt = prompt
                if positive_prompt is not None:
                    entire_prompt = str(prompt) + ' , ' + str(positive_prompt)
                image = pipe(prompt=entire_prompt, height=side_y, width=side_x, num_inference_steps=int(float(steps)), negative_prompt=negative_prompt).images[0]
                del pipe
                torch.cuda.empty_cache()

                return image
            except Exception as e:
                print(f"Attempt {attempt}: Image generation failed with error: {e}")
                del pipe
                torch.cuda.empty_cache()
                if attempt < max_retries:
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError("Maximum retries reached, image generation failed")
    @staticmethod
    def download_from_hf_spaces(repo_id, filename):
        config = AppConfig()
        hf_api_key = config.get_huggingface_api_key()

        if hf_api_key is None:
            raise ValueError("Hugging Face API key not found in configuration")

        # Download the repository locally
        local_repo_dir = snapshot_download(repo_id, token=hf_api_key)

        # Check if the file exists in the downloaded repository
        local_file_path = os.path.join(local_repo_dir, filename)
        if not os.path.exists(local_file_path):
            raise ValueError(f"Error: File {filename} not found in the repository {repo_id} via local dir {local_repo_dir}")

        # Read the file and return its content
        with open(local_file_path, "rb") as f:
            content = f.read()

        return content
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