import os
import datetime
import discord
import asyncio
from threading import Lock
import traceback
from discord.ext import commands
from PIL import Image
from io import BytesIO
from classes.app_config import AppConfig
from classes.image_generator import ImageGenerator
from classes.message_handler import MessageHandler

config = AppConfig()
TOKEN = config.get_discord_api_key()

# Threaded conversation handling
image_queue = asyncio.Queue()
appconfig_lock = asyncio.Lock()
image_queue_lock = Lock()

intents = discord.Intents.default()
intents.typing = False
intents.presences = False

bot = commands.Bot(command_prefix='!', intents=intents)
image_generator = ImageGenerator()
message_handler = MessageHandler(image_generator=image_generator, config=config)
message_handler.set_bot(bot)

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')
    print(f'Server(s) connected: {[guild.name for guild in bot.guilds]}')

async def send_large_message(ctx, text, max_chars=2000):
    if len(text) <= max_chars:
        await ctx.send(text)
        return

    lines = text.split("\n")
    buffer = ""
    first_message = None
    for line in lines:
        if len(buffer) + len(line) + 1 > max_chars:
            if not first_message:
                first_message = await ctx.send(buffer)
                thread = await first_message.create_thread(name="Model List")
            else:
                await thread.send_message(buffer)
            buffer = ""
        buffer += line + "\n"

    if buffer:
        await thread.send_message(buffer)

@bot.command(name='ping', help='Test command to check if the bot is responding.')
async def ping(ctx):
    await ctx.send('Pong!')

@bot.command(name='generate', help='Generates an image based on the given prompt.')
async def generate_image(ctx, *, prompt):
    try:
        await ctx.send(f'Adding prompt to queue for processing: ' + prompt)
        # Put the context and prompt in a tuple before adding it to the queue
        await image_queue.put((ctx, prompt))
        # Run the generate_image_from_queue function as a task, if not already running
        if not hasattr(bot, "image_generation_task") or bot.image_generation_task.done():
            bot.image_generation_task = bot.loop.create_task(generate_image_from_queue())
    except Exception as e:
        await ctx.send(f'Error generating image: {e}\n\nStack trace:\n{traceback.format_exc()}')

# New function to process the prompts in the queue and generate images one at a time
async def generate_image_from_queue():
    while not image_queue.empty():
        ctx, prompt = await image_queue.get()
        await ctx.send("Begin processing queue item: " + prompt)
        user_id = ctx.author.id
        async with appconfig_lock:
            user_config = config.get_user_config(user_id)
            steps = config.get_user_steps(user_id)
            negative_prompt = config.get_user_negative_prompt(user_id=ctx.author.id)
            positive_prompt = config.get_user_positive_prompt(user_id=ctx.author.id)
            resolution = config.get_user_resolution(user_id=ctx.author.id)
            model_id = user_config.get('model', None)
        try:
            # Run the image generation in an executor
            await ctx.send("Begin image generation: " + prompt)
            image = await bot.loop.run_in_executor(None, image_generator.generate_image, prompt, model_id, resolution, negative_prompt, steps, positive_prompt)
            await ctx.send("Begin image processing: " + prompt)
            buffer = BytesIO()
            image.save(buffer, 'PNG')
            buffer.seek(0)
            message = "Prompt: " + str(prompt) + "\nModel: " + str(model_id) + "\nResolution: " + str(resolution['width']) + 'x' + str(resolution['height']) + "\nSteps: " + str(steps)
            await ctx.send(content=message, file=discord.File(buffer, 'generated_image.png'))
        except Exception as e:
            error_message = f'Error generating image: {e}\n\nStack trace:\n{traceback.format_exc()}'
            await send_large_message(ctx, f'Error generating image: {error_message}')
        finally:
            image_queue.task_done()
# Set model command with AppConfig lock
@bot.command(name='setmodel', help='Set the default model for the user.')
async def set_model(ctx, *, model_id = None):
    user_id = ctx.author.id
    async with appconfig_lock:
        user_config = config.get_user_config(user_id)
        if model_id is None:
            model_id = user_config.get('model', None)
            if model_id is None:
                model_id = config.get_default_model()
            await ctx.send(f'Your current model is set to {model_id}.')
            return
        user_config['model'] = model_id
        config.set_user_config(user_id, user_config)
    await ctx.send(f'Default model for user {ctx.author.name} has been set to {model_id}.')
# Set resolution command with AppConfig lock
@bot.command(name='steps', help='Sets the intensity for generated images. Max 1500.')
async def set_steps(ctx, steps = None):
    user_id = ctx.author.id
    async with appconfig_lock:
        if steps is None:
            steps = config.get_user_steps(user_id)
            await ctx.send("Your current steps are set to " + str(steps))
        else:
            old_steps = config.get_user_steps(user_id)
            config.set_user_steps(user_id, steps)
            await ctx.send("Your steps have been updated from " + str(old_steps) + " to " + str(steps))
# Set resolution command with AppConfig lock
available_resolutions = asyncio.run(image_generator.list_available_resolutions())
@bot.command(name='resolution', help='Set or get your default resolution for generated images.\nAvailable resolutions:\n' + str(available_resolutions))
async def set_resolution(ctx, resolution = None):
    try:
        user_id = ctx.author.id
        available_resolutions = await image_generator.list_available_resolutions(user_id, resolution)
        async with appconfig_lock:
            if resolution is None:
                resolution = config.get_user_resolution(user_id)
                await ctx.send(f'Available resolutions:\n' + available_resolutions)
                return
        if 'x' in resolution:
            width, height = map(int, resolution.split('x'))
        else:
            width, height = map(int, resolution.split())

        async with appconfig_lock:
            if available_resolutions is False:
                await ctx.send(f'Invalid resolution. Available resolutions:\n' + await image_generator.list_available_resolutions())
                return
            config.set_user_resolution(user_id, width, height)
            total_pixels = width * height
        await ctx.send(f"Default resolution set to {width}x{height} for user {ctx.author.name} using {total_pixels} pixel count.")
    except ValueError:
        await ctx.send("Invalid resolution format. Please provide resolution as '1920x1080' or '1920 1080'.")

@bot.command(name='listmodels', help='Lists the available models from Hugging Face.')
async def list_models(ctx):
    try:
        models = image_generator.get_available_models()
        models_text = "Available models:\n" + "\n".join(models)
        await send_large_message(ctx, models_text)
    except Exception as e:
        await ctx.send(f'Error retrieving models: {e}\n\nStack trace:\n{traceback.format_exc()}')

# Negative prompt command with AppConfig lock
@bot.command(name='negative', help='Gets or sets the negative prompt.')
async def negative(ctx, *, negative_prompt = None):
    try:
        async with appconfig_lock:
            if negative_prompt is None:
                negative_prompt = config.get_user_negative_prompt(user_id=ctx.author.id)
                output_text = "Your negative prompt currently is set as:\n" + negative_prompt
            else:
                config.set_user_negative_prompt(user_id=ctx.author.id, negative_prompt=negative_prompt)
                output_text = "Your negative prompt is now set to:\n" + negative_prompt
        await send_large_message(ctx, output_text)
    except Exception as e:
        await ctx.send(f'Error running negative prompt handler: {e}\n\nStack trace:\n```{traceback.format_exc()}```')

# Positive prompt command with AppConfig lock
@bot.command(name='positive', help='Gets or sets the positive POST-prompt. A value of "none" disables this. It is added to the end of every prompt you submit via !generate.')
async def positive(ctx, *, positive_prompt = None):
    try:
        async with appconfig_lock:
            if positive_prompt is None:
                positive_prompt = config.get_user_positive_prompt(user_id=ctx.author.id)
                output_text = "Your positive prompt currently is set as:\n" + positive_prompt
            elif positive_prompt == "none":
                config.set_user_positive_prompt(user_id=ctx.author.id, positive_prompt="")
                output_text = "Your positive prompt is now set to:\n" + positive_prompt
            else:
                config.set_user_positive_prompt(user_id=ctx.author.id, positive_prompt=positive_prompt)
                output_text = "Your positive prompt is now set to:\n" + positive_prompt
        await send_large_message(ctx, output_text)
    except Exception as e:
        await ctx.send(f'Error running negative prompt handler: {e}\n\nStack trace:\n{traceback.format_exc()}')

## Image queue management
def is_server_admin(ctx):
    admin_role_name = "Image Admin"  # Replace this with the role name used on your server for admins
    return any(role.name == admin_role_name for role in ctx.author.roles)
@bot.command(name='listqueue', help='Lists the contents of the image generation queue (Admin only).')
async def list_queue(ctx):
    if not is_server_admin(ctx):
        await ctx.send("You must be a server admin to use this command.")
        return

    if image_queue.empty():
        await ctx.send("The image generation queue is currently empty.")
        return

    queue_contents = [f"{index}: {item[1]}" for index, item in enumerate(image_queue._queue)]
    await send_large_message(ctx, "Image generation queue:\n" + "\n".join(queue_contents))

@bot.command(name='removequeue', help='Removes an entry from the image generation queue by index (Admin only).')
async def remove_queue(ctx, index: int):
    if not is_server_admin(ctx):
        await ctx.send("You must be a server admin to use this command.")
        return
    global image_queue

    new_queue, removed_item = await remove_item_from_queue(image_queue, index)
    if removed_item is None:
        await ctx.send("Invalid queue index. Please provide a valid index.")
        return

    image_queue = new_queue
    await ctx.send(f"Removed item '{removed_item[1]}' from the image generation queue.")
async def remove_item_from_queue(queue, index):
    if index < 0 or index >= queue.qsize():
        return None
    new_queue = asyncio.Queue()
    removed_item = None
    for i in range(queue.qsize()):
        item = await queue.get()
        if i == index:
            removed_item = item
        else:
            await new_queue.put(item)

    return new_queue, removed_item
async def generate_variants(image_generator, input_image):
    prompt = "a variation of the input image"
    generated_image = image_generator.generate_image(prompt, model_id="lambdalabs/sd-image-variations-diffusers", steps=25)
    return [generated_image]
@bot.event
async def on_message(message):
    # If the message is from the bot itself, ignore it
    if message.author == bot.user:
        return
    await message_handler.handle_message(message)
if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Starting bot...")
    bot.run(TOKEN)
