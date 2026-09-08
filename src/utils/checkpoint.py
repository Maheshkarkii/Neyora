import os
import torch
from typing import Dict, Any, Optional
from src.utils.logger import get_logger

logger = get_logger("checkpoint_utils")

def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str = "checkpoints",
    filename: str = "best_translation_model.pt"
) -> str:
    """
    Saves model state, optimizer state, epoch, validation loss, and configuration.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    logger.info(f"Saved checkpoint to {filepath} (Epoch {state.get('epoch')}, Val Loss: {state.get('val_loss', 0.0):.4f})")
    return filepath

def load_checkpoint(
    filepath: str,
    map_location: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Loads a checkpoint dictionary from disk.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at: {filepath}")
    checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)
    logger.info(f"Loaded checkpoint from {filepath}")
    return checkpoint
