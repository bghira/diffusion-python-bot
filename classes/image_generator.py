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
from transformers import AutoModel

# Set up the logging configuration
logging.basicConfig(level=logging.INFO)

class ImageGenerator:
    def __init__(self, device="cuda", torch_dtype=torch.float16):
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype

    def get_variation_pipe(self, model_id, use_attention_scaling = False):
        logging.info("Clearing the CUDA cache...")
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        logging.info("Generating a new variation pipe...")
        pipe = StableDiffusionImageVariationPipeline.from_pretrained(pretrained_model_name_or_path=model_id, torch_dtype=self.torch_dtype)
        if use_attention_scaling:
            logging.info('Using attention scaling, because high resolution was selected! Safety first!!')
            pipe.enable_sequential_cpu_offload()
            pipe.enable_attention_slicing(1)
            # torch.backends.cudnn.benchmark = True
            # torch.backends.cudnn.enabled = True
        else:
            pipe.to(self.device)
        pipe.safety_checker = lambda images, clip_input: (images, False)
        logging.info("Return the pipe...")
        return pipe
    def get_pipe(self, model_id, use_attention_scaling = False):
        logging.info("Clearing the CUDA cache...")
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        logging.info("Generating a new pipe...")
        if use_attention_scaling is False:
            pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path=model_id, torch_dtype=self.torch_dtype)
            pipe.to(self.device)
        elif use_attention_scaling:
            logging.info('Using attention scaling, because high resolution was selected! Safety first!!')
            pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path=model_id, load_in_8bit=True)
            pipe.enable_sequential_cpu_offload()
            pipe.enable_attention_slicing(1)
            # torch.backends.cudnn.benchmark = True
            # torch.backends.cudnn.enabled = True
        pipe.safety_checker = lambda images, clip_input: (images, False)
        logging.info("Return the pipe...")
        return pipe

    def generate_image_variations(self, width, height, input_image, num_inference_steps=25, guidance_scale=7.5, use_attention_scaling = False):
        input_image = input_image.resize((512, 384))
        logging.info("Initializing image variation generation pipeline...")
        total_pixel_count = width * height
        if total_pixel_count > 393216:
            estimated_time = -2.492e-6 * total_pixel_count + 3.4
            logging.info('Resolution ' + str(width) + 'x' + str(height) + ' has a pixel count greater than threshold. Using attention scaling expects to take '+ str(estimated_time) + ' seconds.')
            use_attention_scaling = True
        pipe = self.get_variation_pipe("lambdalabs/sd-image-variations-diffusers", use_attention_scaling=use_attention_scaling)
        input_image = pad(input_image, (input_image.size[0] // 2, input_image.size[1] // 2))
        # Generate image variations
        generated_images = pipe(width=width, height=height, image=input_image, guidance_scale=guidance_scale, num_inference_steps=int(float(num_inference_steps))).images
        del pipe
        return generated_images

    def generate_image(self, prompt, model_id=None, resolution=None, negative_prompt=None, steps=50, positive_prompt=None, max_retries=1, retry_delay=10):
        logging.info("Initializing image generation pipeline...")
        use_attention_scaling = False
        if resolution is not None:
            total_pixel_count = resolution['width'] * resolution['height']
            if total_pixel_count > 393216:
                estimated_time = -2.492e-6 * total_pixel_count + 3.4
                logging.info('Resolution ' + str(resolution['width']) + 'x' + str(resolution['height']) + ' has a pixel count greater than threshold. Using attention scaling expects to take '+ str(estimated_time) + ' seconds.')
                use_attention_scaling = True
            side_x = resolution['width']
            side_y = resolution['height']
        else:
            side_x = side_y = None
        logging.info("Set custom resolution")
        pipe = self.get_pipe(model_id, use_attention_scaling)
        logging.info("Copied pipe to the local context")
        # Combine the main prompt and positive_prompt if provided
        entire_prompt = prompt
        if positive_prompt is not None:
            entire_prompt = str(prompt) + ' , ' + str(positive_prompt)
        for attempt in range(1, max_retries + 1):
            try:
                logging.info(f"Attempt {attempt}: Generating image...")
                with torch.no_grad():
                    image = pipe(prompt=entire_prompt, height=side_y, width=side_x, num_inference_steps=int(float(steps)), negative_prompt=negative_prompt).images[0]
                image = image.resize((1920, 1080))

                del pipe
                import gc
                gc.collect()
                torch.cuda.empty_cache()

                logging.info("Image generation successful!")
                return image
            except Exception as e:
                logging.error(f"Attempt {attempt}: Image generation failed with error: {e}")
                del pipe
                import gc
                gc.collect()
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
    # Helper to list / check a given resolution against the fold.
    async def list_available_resolutions(self, user_id = None, resolution=None):
        config = AppConfig()
        resolutions = [
            {'width': 16, 'height': 16},
            {'width': 24, 'height': 16},
            {'width': 32, 'height': 24},
            {'width': 48, 'height': 32},
            {'width': 64, 'height': 48},
            {'width': 96, 'height': 64},
            {'width': 128, 'height': 96},
            {'width': 192, 'height': 128},
            {'width': 256, 'height': 192},
            {'width': 384, 'height': 256},
            {'width': 512, 'height': 384},
            {'width': 768, 'height': 512},
            {'width': 800, 'height': 456},
            {'width': 1024, 'height': 576},
            {'width': 1152, 'height': 648},
            {'width': 1280, 'height': 720},
            # Add more resolutions if needed
        ]
        current_resolution_indicator = " "
        if resolution is not None:
            width, height = map(int, resolution.split('x'))
            if any(r['width'] == width and r['height'] == height for r in resolutions):
                return True
            else:
                return False

        resolution_list = ""
        for r in resolutions:
            if user_id is not None:
                user_resolution = config.get_user_resolution(user_id=user_id)
                if user_resolution is not None:
                    if user_resolution['width'] == r['width'] and user_resolution['height'] == r['height']:
                        current_resolution_indicator = "\>"
            resolution_list += "  " + current_resolution_indicator + " " + f"{r['width']}x{r['height']}\n"
            current_resolution_indicator = "  "

        return resolution_list
