import yaml
import os
from pathlib import Path

def load_prompts(path: Path) -> dict:
    print(f"---LOADING PROMPTS FROM {path}---")
    with open(path, 'r') as file:
        prompts = yaml.safe_load(file)
    return prompts