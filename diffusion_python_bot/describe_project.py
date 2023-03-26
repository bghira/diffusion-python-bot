import os
import openai
from pathlib import Path
from classes.app_config import AppConfig


# Function to traverse the project and read content
def traverse_project(path):
    content = ''
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.py'):
                file_path = Path(root) / filename
                with open(file_path, 'r') as file:
                    content += file.read()
    return content


# Function to tokenize the content (use your preferred tokenizer)
def tokenize(content):
    # TODO -- choose and get tokenizer, NLTK seems like a good first choice
    return content


# Set your API key from OpenAI
config = AppConfig()
TOKEN = config.get_discord_api_key()


# Function to send tokenized text to GPT-3 API and get TL;DR
def get_tldr(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Create a TL;DR for the following Python project:\n{prompt}",
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].text.strip()


if __name__ == "__main__":
    project_path = "/diffusion_python_bot"
    project_content = traverse_project(project_path)
    tokenized_content = tokenize(project_content)

    tldr = get_tldr(tokenized_content)
    print(f"TL;DR: {tldr}")