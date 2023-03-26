class DiscordProgressBar:
    def __init__(self, ctx, total_steps, original_stdout, progress_bar_length=20):
        self.ctx = ctx
        self.total_steps = total_steps
        self.progress_bar_length = progress_bar_length
        self.original_stdout = original_stdout
        self.current_step = 0
        print("Created discord progress bar tracker.")

    async def update_progress_bar(self, step):
        print("Running discord update_progress_bar")
        print("update_progress_bar hit step " + str(step))
        self.current_step = step
        progress = self.current_step / self.total_steps
        filled_length = int(progress * self.progress_bar_length)
        bar = "█" * filled_length + "-" * (self.progress_bar_length - filled_length)
        percent = round(progress * 100, 1)
        progress_text = f"[{bar}] {percent}% complete"
        print(f"Updating progress in Discord: {progress_text}\n")  # Add this line
        await self.ctx.send(progress_text)