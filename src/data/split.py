import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from src.utils.reproducibility import set_seed
from src.utils.logger import get_logger

logger = get_logger("data_split")

def create_asr_splits(
    records: List[Dict[str, Any]],
    output_dir: str = "data/metadata",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Creates speaker-disjoint train, validation, and test splits for ASR.
    
    Why Speaker-Independent Evaluation Matters:
    If utterances from the same speaker appear in both the training and test sets (speaker overlap),
    the ASR model can overfit to specific speaker characteristics (vocal tract length, pitch, accent,
    recording room acoustics) rather than learning invariant speech-to-grapheme mappings.
    Disjoint speaker partitioning provides a realistic benchmark of generalization to unseen speakers.
    """
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.DataFrame(records)
    speakers = sorted(df["speaker_id"].unique())
    np.random.shuffle(speakers)
    
    n_train_spk = max(1, int(len(speakers) * train_ratio))
    n_val_spk = max(1, int(len(speakers) * val_ratio))
    
    train_spks = set(speakers[:n_train_spk])
    val_spks = set(speakers[n_train_spk:n_train_spk + n_val_spk])
    test_spks = set(speakers[n_train_spk + n_val_spk:])
    if not test_spks:
        test_spks = val_spks

    train_df = df[df["speaker_id"].isin(train_spks)].copy()
    val_df = df[df["speaker_id"].isin(val_spks)].copy()
    test_df = df[df["speaker_id"].isin(test_spks)].copy()

    train_path = os.path.join(output_dir, "asr_train.csv")
    val_path = os.path.join(output_dir, "asr_val.csv")
    test_path = os.path.join(output_dir, "asr_test.csv")

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    logger.info(f"ASR Speaker Splits Saved: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return {"train": train_df, "val": val_df, "test": test_df}


def create_translation_splits(
    pairs: List[Dict[str, str]],
    output_dir: str = "data/metadata",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Creates disjoint, deduplicated train, validation, and test splits for NMT.
    """
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(pairs)
    # Deduplicate on source text to prevent test data leakage
    df = df.drop_duplicates(subset=["nepali_text"]).reset_index(drop=True)

    indices = np.arange(len(df))
    np.random.shuffle(indices)

    n_train = max(1, int(len(indices) * train_ratio))
    n_val = max(1, int(len(indices) * val_ratio))

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    if len(test_idx) == 0:
        test_idx = val_idx

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    train_path = os.path.join(output_dir, "translation_train.csv")
    val_path = os.path.join(output_dir, "translation_val.csv")
    test_path = os.path.join(output_dir, "translation_test.csv")

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    logger.info(f"Translation Splits Saved: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return {"train": train_df, "val": val_df, "test": test_df}
