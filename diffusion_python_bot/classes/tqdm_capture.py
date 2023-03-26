import re, asyncio
from .discord_progress_bar import DiscordProgressBar

class TqdmCapture:
    def __init__(self, progress_bar: DiscordProgressBar, loop, original_stdout, original_stderr):
        self.progress_bar = progress_bar
        self.loop = loop
        self.original_stdout = original_stdout
        self.original_stderr = original_stderr

    def write(self, s: str):
        test_string = s.strip()
        if test_string != "":
            self.original_stderr.write(f"tqdm stderrput: \"" + s.strip() + "\"\n")  # Add this line
            match = re.search(r'\b(\d+)%\|', s)
            if match:
                progress = int(match.group(1))
                self.original_stdout.write(f"Captured progress: {progress}\n")  # Add this line
                asyncio.run_coroutine_threadsafe(self.progress_bar.update_progress_bar(progress), self.loop)

    def flush(self):
        pass