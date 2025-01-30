import json

def load_config(config_file='config.json'):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config
