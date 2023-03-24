# classes/app_config.py

import json
import os
from io import BytesIO

class AppConfig:
    def __init__(self):
        self.config_path = "config/config.json"
        self.example_config_path = "config/example.json"

        if not os.path.exists(self.config_path):
            with open(self.example_config_path, "r") as example_file:
                example_config = json.load(example_file)

            with open(self.config_path, "w") as config_file:
                json.dump(example_config, config_file)

        with open(self.config_path, "r") as config_file:
            self.config = json.load(config_file)

    def get_discord_api_key(self):
        return self.config['discord_api']['api_key']
    def get_huggingface_api_key(self):
        return self.config['huggingface_api'].get('api_key', None)

    def get_local_model_path(self):
        return self.config['huggingface'].get('local_model_path', None)
    def get_user_config(self, user_id):
        return self.config['users'].get(str(user_id), {})
    def set_user_config(self, user_id, user_config):
        self.config['users'][str(user_id)] = user_config
        with open(self.config_path, "w") as config_file:
            json.dump(self.config, config_file)
    def set_user_positive_prompt(self, user_id, positive_prompt):
        user_key = f"user_{user_id}"
        if user_key not in self.config:
            self.config[user_key] = {}
        self.config[user_key]['positive_prompt'] = positive_prompt
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4) 
    def set_user_negative_prompt(self, user_id, negative_prompt):
        user_key = f"user_{user_id}"
        if user_key not in self.config:
            self.config[user_key] = {}
        self.config[user_key]['negative_prompt'] = negative_prompt
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4) 
    def set_user_steps(self, user_id, steps: int):
        user_key = f"user_{user_id}"
        if user_key not in self.config:
            self.config[user_key] = {}
        self.config[user_key]['steps'] = steps
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4) 
    def set_user_resolution(self, user_id, width, height):
        user_key = f"user_{user_id}"
        if user_key not in self.config:
            self.config[user_key] = {}
        
        self.config[user_key]['resolution'] = {
            'width': width,
            'height': height
        }

        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)

    def get_user_positive_prompt(self, user_id):
        user_key = f"user_{user_id}"
        return self.config.get(user_key, {}).get('positive_prompt', 'beautiful hyperrealistic')

    def get_user_negative_prompt(self, user_id):
        user_key = f"user_{user_id}"
        return self.config.get(user_key, {}).get('negative_prompt', '(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, flowers, human, man, woman')

    def get_user_resolution(self, user_id):
        user_key = f"user_{user_id}"
        return self.config.get(user_key, {}).get('resolution', None)
    def get_user_steps(self, user_id):
        user_key = f"user_{user_id}"
        return self.config.get(user_key, {}).get('steps', 250)
