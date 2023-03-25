import discord
import traceback
from io import BytesIO
from .image_generator import ImageGenerator
from .app_config import ( AppConfig as config, AppConfig )
import logging
from PIL import Image


class MessageHandler:
    def __init__(self, image_generator: ImageGenerator, config: AppConfig):
        self.image_generator = image_generator
        self.config = config
        self.bot = None  # This will be set later

    def set_bot(self, bot):
        self.bot = bot

    async def handle_message(self, message):
        if message.attachments:
            await self._handle_image_attachment(message)
        await self.bot.process_commands(message)

    async def _handle_image_attachment(self, message):
        user_id = message.author.id
        logging.info("User id: " + str(user_id))
        resolution = self.config.get_user_resolution(user_id=message.author.id)
        num_inference_steps = self.config.get_user_steps(user_id=message.author.id)
        width = resolution['width']
        height = resolution['height']
        for attachment in message.attachments:
            if attachment.content_type.startswith('image/'):
                image_data = await attachment.read()
                input_image = Image.open(BytesIO(image_data))
                prompt = message.content
                try:
                    generated_images = await self.bot.loop.run_in_executor(None, self.image_generator.generate_image_variations, width, height, input_image, num_inference_steps)
                    for i, image in enumerate(generated_images):
                        buffer = BytesIO()
                        image.resize({1920, 1080}).save(buffer, 'PNG')
                        buffer.seek(0)
                        await message.channel.send(file=discord.File(buffer, f'variant_{i}.png'))
                except Exception as e:
                    error_message = f'Error generating image variant: {e}\n\nStack trace:\n{traceback.format_exc()}'
                    await message.channel.send(error_message)
    async def send_large_message(self, ctx, text, max_chars=2000):
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
    