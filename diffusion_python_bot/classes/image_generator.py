# classes/image_generator.py
import os
import sys
import logging

# Tell TensorFlow to be quiet.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Import the library
import torch
from torchvision.transforms.functional import pad
from diffusers import StableDiffusionPipeline, StableDiffusionImageVariationPipeline

import time
from .app_config import AppConfig
from asyncio import Lock
from tqdm import tqdm
import traceback


class ImageGenerator:
    resolutions = [
        {"width": 128, "height": 96, "scaling_factor": 100},
        {"width": 192, "height": 128, "scaling_factor": 94},
        {"width": 256, "height": 192, "scaling_factor": 88},
        {"width": 384, "height": 256, "scaling_factor": 76},
        {"width": 512, "height": 384, "scaling_factor": 64},
        {"width": 768, "height": 512, "scaling_factor": 52},
        {"width": 800, "height": 456, "scaling_factor": 50},
        {"width": 1024, "height": 576, "scaling_factor": 40},
        {"width": 1152, "height": 648, "scaling_factor": 34},
        {"width": 1280, "height": 720, "scaling_factor": 30},
        # Add more resolutions if needed
    ]

    def __init__(
        self, shared_queue_lock: Lock, device="cuda", torch_dtype=torch.float16
    ):
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.lock = shared_queue_lock

    def get_variation_pipe(self, model_id, use_attention_scaling=False):
        logging.info("Clearing the CUDA cache...")
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        logging.info("Generating a new variation pipe...")
        pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            pretrained_model_name_or_path=model_id, torch_dtype=self.torch_dtype
        )
        logging.info(
            "Using attention scaling, because a variation is being crafted! This will make generation run more slowly, but it will be less likely to run out of memory."
        )
        pipe.enable_sequential_cpu_offload()
        pipe.enable_attention_slicing(1)
        # torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.enabled = True
        pipe.safety_checker = lambda images, clip_input: (images, False)
        logging.info("Return the pipe...")
        return pipe

    def get_pipe(self, model_id, use_attention_scaling=False):
        logging.info("Clearing the CUDA cache...")
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        logging.info("Generating a new pipe...")
        if use_attention_scaling is False:
            pipe = StableDiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path=model_id, torch_dtype=self.torch_dtype
            )
            pipe.to(self.device)
        elif use_attention_scaling:
            logging.info(
                "Using attention scaling, because high resolution was selected! Safety first!!"
            )
            pipe = StableDiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path=model_id, load_in_8bit=True
            )
            pipe.enable_sequential_cpu_offload()
            pipe.enable_attention_slicing(1)
            # torch.backends.cudnn.benchmark = True
            # torch.backends.cudnn.enabled = True
        pipe.safety_checker = lambda images, clip_input: (images, False)
        logging.info("Return the pipe...")
        return pipe

    def generate_image_variations(
        self,
        width: int,
        height: int,
        input_image,
        steps: int,
        tqdm_capture,
        guidance_scale=7.5,
        use_attention_scaling=False,
    ):
        input_image = input_image.resize((512, 384))
        logging.info("Initializing image variation generation pipeline...")
        scaling_factor = self.get_scaling_factor(width, height, self.resolutions)
        if int(steps) > int(scaling_factor):
            steps = int(scaling_factor)
        logging.info(f"Scaling factor for {width}x{height}: {scaling_factor}")
        if scaling_factor < 50:
            logging.info(
                "Resolution "
                + str(width)
                + "x"
                + str(height)
                + " has a pixel count greater than threshold. Using attention scaling expects to take 30 seconds."
            )
            use_attention_scaling = True
        pipe = self.get_variation_pipe(
            "lambdalabs/sd-image-variations-diffusers",
            use_attention_scaling=use_attention_scaling,
        )
        input_image = pad(
            input_image, (input_image.size[0] // 2, input_image.size[1] // 2)
        )
        # Generate image variations
        with tqdm(total=steps, ncols=100, file=tqdm_capture) as pbar:
            generated_images = pipe(
                width=width,
                height=height,
                image=input_image,
                guidance_scale=guidance_scale,
                num_inference_steps=int(float(steps)),
            ).images
        del pipe
        return generated_images

    def generate_image(
        self,
        prompt,
        model_id,
        resolution,
        negative_prompt,
        steps,
        positive_prompt,
        tqdm_capture,
    ):
        logging.info("Initializing image generation pipeline...")
        use_attention_scaling = False
        max_retries = retry_delay = 5
        if resolution is not None:
            scaling_factor = self.get_scaling_factor(
                resolution["width"], resolution["height"], self.resolutions
            )
            logging.info(
                f"Scaling factor for {resolution['width']}x{resolution['height']}: {scaling_factor}"
            )
            if scaling_factor < 50:
                logging.info(
                    "Resolution "
                    + str(resolution["width"])
                    + "x"
                    + str(resolution["height"])
                    + " has a pixel count greater than threshold. Using attention scaling expects to take 30 seconds."
                )
                use_attention_scaling = True
                if steps > scaling_factor:
                    steps = scaling_factor
            side_x = resolution["width"]
            side_y = resolution["height"]
        else:
            side_x = side_y = None
        logging.info("Set custom resolution")
        pipe = self.get_pipe(model_id, use_attention_scaling)
        logging.info("Copied pipe to the local context")
        logging.info("REDIRECTING THE PRECIOUS, STDOUT... SORRY IF THAT UPSETS YOU")
        # Redirect sys.stdout to capture tqdm output
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        # sys.stdout = tqdm_capture
        # sys.stderr = FilteringStderrWrapper(sys.stderr, sys.stdout)
        sys.stderr = tqdm_capture
        # Combine the main prompt and positive_prompt if provided
        entire_prompt = prompt
        if positive_prompt is not None:
            entire_prompt = str(prompt) + " , " + str(positive_prompt)
        for attempt in range(1, max_retries + 1):
            try:
                logging.info(f"Attempt {attempt}: Generating image...")
                with torch.no_grad():
                    with tqdm(total=steps, ncols=100, file=tqdm_capture) as pbar:
                        image = pipe(
                            prompt=entire_prompt,
                            height=side_y,
                            width=side_x,
                            num_inference_steps=int(float(steps)),
                            negative_prompt=negative_prompt,
                        ).images[0]
                image = image.resize((1920, 1080))

                del pipe
                import gc

                gc.collect()
                torch.cuda.empty_cache()

                logging.info("Image generation successful!")
                return image
            except Exception as e:
                logging.error(
                    f"Error generating image: {e}\n\nStack trace:\n{traceback.format_exc()}"
                )
                del pipe
                import gc

                gc.collect()
                torch.cuda.empty_cache()
                if attempt < max_retries:
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError(
                        "Maximum retries reached, image generation failed"
                    )
            finally:
                # Don't forget to restore the original stdout after the image generation is done
                # sys.stdout = original_stdout
                sys.stderr = original_stderr

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
                    model_id = subdir[len(model_prefix) :].replace("--", "/")
                    local_models.append(model_id)

        return local_models

    # Helper to list / check a given resolution against the fold.
    async def list_available_resolutions(self, user_id=None, resolution=None):
        config = AppConfig()

        current_resolution_indicator = " "
        if resolution is not None:
            width, height = map(int, resolution.split("x"))
            if any(
                r["width"] == width and r["height"] == height for r in self.resolutions
            ):
                return True
            else:
                return False

        resolution_list = ""
        for r in self.resolutions:
            if user_id is not None:
                user_resolution = config.get_user_setting(
                    user_id, "resolution", {"width": 800, "height": 456}
                )
                if user_resolution is not None:
                    if (
                        user_resolution["width"] == r["width"]
                        and user_resolution["height"] == r["height"]
                    ):
                        current_resolution_indicator = "\>"
            resolution_list += (
                "  "
                + current_resolution_indicator
                + " "
                + f"{r['width']}x{r['height']}\n"
            )
            current_resolution_indicator = "  "

        return resolution_list

    def get_scaling_factor(self, width, height, scaled_resolutions):
        for res in scaled_resolutions:
            if res["width"] == width and res["height"] == height:
                return int(res["scaling_factor"])
        return None

    def is_valid_resolution(self, width, height):
        for res in self.resolutions:
            if res["width"] == width and res["height"] == height:
                return True
        return False
