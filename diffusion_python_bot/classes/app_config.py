# classes/app_config.py

import json
import os
import sys
from io import BytesIO


class AppConfig:
    def __init__(self):
        from pathlib import Path

        parent = os.path.dirname(Path(__file__).resolve().parent)
        config_path = os.path.join(parent, "config")
        self.config_path = os.path.join(config_path, "config.json")
        self.example_config_path = os.path.join(config_path, "example.json")

        if not os.path.exists(self.config_path):
            with open(self.example_config_path, "r") as example_file:
                example_config = json.load(example_file)

            with open(self.config_path, "w") as config_file:
                json.dump(example_config, config_file)

        with open(self.config_path, "r") as config_file:
            self.config = json.load(config_file)
    def get_concurrent_slots(self):
        return self.config.get("concurrent_slots", 1)
    def get_discord_api_key(self):
        return self.config["discord_api"]["api_key"]

    def get_huggingface_api_key(self):
        return self.config["huggingface_api"].get("api_key", None)

    def get_local_model_path(self):
        return self.config["huggingface"].get("local_model_path", None)

    def get_user_config(self, user_id):
        return self.config["users"].get(str(user_id), {})

    def set_user_config(self, user_id, user_config):
        self.config["users"][str(user_id)] = user_config
        with open(self.config_path, "w") as config_file:
            json.dump(self.config, config_file)

    def set_user_setting(self, user_id, setting_key, value):
        user_id = str(user_id)
        if user_id not in self.config["users"]:
            self.config["users"][user_id] = {}
        self.config["users"][user_id][setting_key] = value
        with open(self.config_path, "w") as config_file:
            json.dump(self.config, config_file)

    def get_user_setting(self, user_id, setting_key, default_value=None):
        user_id = str(user_id)
        return self.config["users"].get(user_id, {}).get(setting_key, default_value)
