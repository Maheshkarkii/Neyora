import random
import os
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """
    Ensures deterministic reproducibility across Python, NumPy, PyTorch CPU, and CUDA.
    
    Documented Seed Configuration:
    - Python `random.seed(seed)`
    - OS Environment `PYTHONHASHSEED = str(seed)`
    - NumPy `np.random.seed(seed)`
    - PyTorch `torch.manual_seed(seed)`
    - PyTorch CUDA `torch.cuda.manual_seed_all(seed)`
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def seed_worker(worker_id: int) -> None:
    """DataLoader worker initializer for multi-process data loading reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
