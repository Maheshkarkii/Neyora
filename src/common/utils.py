import os
import json
import random
import yaml
import torch
import numpy as np
from typing import Any, Dict, List

def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds for PyTorch, NumPy, and Python."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml_config(config: Dict[str, Any], config_path: str) -> None:
    """Saves a dictionary as a YAML configuration file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

def load_json_manifest(json_path: str) -> List[Dict[str, Any]]:
    """Loads a JSON manifest."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Manifest not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json_manifest(data: List[Dict[str, Any]], json_path: str) -> None:
    """Saves data as a structured JSON file."""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
