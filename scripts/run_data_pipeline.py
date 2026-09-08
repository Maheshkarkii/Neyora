import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.common.logger import setup_logger
from src.common.utils import load_yaml_config, set_seed
from src.asr.download_openslr54 import build_asr_dataset
from src.asr.tokenizer import ASRTokenizer
from src.asr.audio_transforms import AudioPreprocessor
from src.asr.dataset import get_asr_dataloader
from src.nmt.download_parallel_data import build_nmt_dataset
from src.nmt.tokenizer import NMTTokenizer
from src.nmt.dataset import get_nmt_dataloader

logger = setup_logger("data_pipeline_runner")

def run_pipeline():
    set_seed(42)
    logger.info("==========================================================")
    logger.info("   NEPALI VOICE TRANSLATOR - DATA PIPELINE EXECUTION     ")
    logger.info("==========================================================")

    # ---------------------------------------------------------
    # 1. ASR DATA PIPELINE
    # ---------------------------------------------------------
    logger.info("--> [1/2] Processing ASR Audio & Transcripts Pipeline...")
    asr_config = load_yaml_config("configs/asr_config.yaml")
    
    asr_stats = build_asr_dataset(
        raw_dir=asr_config["data"]["raw_dir"],
        processed_dir=asr_config["data"]["processed_dir"],
        sample_limit=asr_config["data"]["dev_sample_limit"],
        target_sample_rate=asr_config["data"]["target_sample_rate"],
        train_split=asr_config["data"]["train_split"],
        val_split=asr_config["data"]["val_split"],
        test_split=asr_config["data"]["test_split"],
        seed=asr_config["data"]["random_seed"]
    )
    
    from src.common.utils import load_json_manifest
    train_asr_data = load_json_manifest(os.path.join(asr_config["data"]["processed_dir"], "train_manifest.json"))
    asr_tokenizer = ASRTokenizer()
    asr_tokenizer.build_vocab_from_texts([d["text"] for d in train_asr_data])
    asr_vocab_path = os.path.join(asr_config["data"]["processed_dir"], "vocab.json")
    asr_tokenizer.save_vocab(asr_vocab_path)
    logger.info(f"Saved ASR Devanagari Vocab (size={asr_tokenizer.vocab_size}) to {asr_vocab_path}")

    preprocessor = AudioPreprocessor(
        target_sample_rate=asr_config["data"]["target_sample_rate"],
        n_mels=asr_config["features"]["n_mels"],
        n_fft=asr_config["features"]["n_fft"],
        hop_length=asr_config["features"]["hop_length"],
        win_length=asr_config["features"]["win_length"]
    )
    
    train_asr_loader = get_asr_dataloader(
        manifest_path=os.path.join(asr_config["data"]["processed_dir"], "train_manifest.json"),
        preprocessor=preprocessor,
        tokenizer=asr_tokenizer,
        batch_size=asr_config["dataloader"]["batch_size"],
        shuffle=True
    )
    val_asr_loader = get_asr_dataloader(
        manifest_path=os.path.join(asr_config["data"]["processed_dir"], "val_manifest.json"),
        preprocessor=preprocessor,
        tokenizer=asr_tokenizer,
        batch_size=asr_config["dataloader"]["batch_size"],
        shuffle=False
    )
    test_asr_loader = get_asr_dataloader(
        manifest_path=os.path.join(asr_config["data"]["processed_dir"], "test_manifest.json"),
        preprocessor=preprocessor,
        tokenizer=asr_tokenizer,
        batch_size=asr_config["dataloader"]["batch_size"],
        shuffle=False
    )

    specs, targets, in_lens, tgt_lens, texts = next(iter(train_asr_loader))
    logger.info(f"ASR Train Batch: Mel Specs Shape = {list(specs.shape)} ({specs.dtype})")
    logger.info(f"ASR Train Batch: Targets Shape   = {list(targets.shape)} ({targets.dtype})")
    logger.info(f"ASR Train Batch: Input Lengths   = {in_lens.tolist()[:4]}...")
    logger.info(f"ASR Train Batch: Target Lengths  = {tgt_lens.tolist()[:4]}...")
    logger.info(f"ASR Train Batch: Sample Text     = '{texts[0]}'")
    logger.info(f"ASR Train Batch: Decoded Text    = '{asr_tokenizer.decode(targets[0].tolist())}'")

    # ---------------------------------------------------------
    # 2. NMT DATA PIPELINE
    # ---------------------------------------------------------
    logger.info("--> [2/2] Processing Nepali-English Translation Pipeline...")
    nmt_config = load_yaml_config("configs/nmt_config.yaml")

    nmt_stats = build_nmt_dataset(
        raw_dir=nmt_config["data"]["raw_dir"],
        processed_dir=nmt_config["data"]["processed_dir"],
        sample_limit=nmt_config["data"]["dev_sample_limit"],
        train_split=nmt_config["data"]["train_split"],
        val_split=nmt_config["data"]["val_split"],
        test_split=nmt_config["data"]["test_split"],
        seed=nmt_config["data"]["random_seed"]
    )

    src_tokenizer = NMTTokenizer.load_vocab(os.path.join(nmt_config["data"]["processed_dir"], "src_vocab.json"))
    tgt_tokenizer = NMTTokenizer.load_vocab(os.path.join(nmt_config["data"]["processed_dir"], "tgt_vocab.json"))

    train_nmt_loader = get_nmt_dataloader(
        json_path=os.path.join(nmt_config["data"]["processed_dir"], "train.json"),
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        batch_size=nmt_config["dataloader"]["batch_size"],
        shuffle=True
    )
    val_nmt_loader = get_nmt_dataloader(
        json_path=os.path.join(nmt_config["data"]["processed_dir"], "val.json"),
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        batch_size=nmt_config["dataloader"]["batch_size"],
        shuffle=False
    )
    test_nmt_loader = get_nmt_dataloader(
        json_path=os.path.join(nmt_config["data"]["processed_dir"], "test.json"),
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        batch_size=nmt_config["dataloader"]["batch_size"],
        shuffle=False
    )

    src_batch, src_lens, tgt_batch, tgt_lens, raw_srcs, raw_tgts = next(iter(train_nmt_loader))
    logger.info(f"NMT Train Batch: Source Shape = {list(src_batch.shape)} ({src_batch.dtype})")
    logger.info(f"NMT Train Batch: Target Shape = {list(tgt_batch.shape)} ({tgt_batch.dtype})")
    logger.info(f"NMT Train Batch: Source Lengths = {src_lens.tolist()[:4]}...")
    logger.info(f"NMT Train Batch: Target Lengths = {tgt_lens.tolist()[:4]}...")
    logger.info(f"NMT Sample Pair: '{raw_srcs[0]}' -> '{raw_tgts[0]}'")
    logger.info(f"NMT Decoded:     '{src_tokenizer.decode(src_batch[0].tolist())}' -> '{tgt_tokenizer.decode(tgt_batch[0].tolist())}'")

    print("\n" + "="*70)
    print("                      DATA PIPELINE SUMMARY                        ")
    print("="*70)
    print(f"ASR Total Audio Samples: {asr_stats['total_samples']} (Train: {len(train_asr_loader.dataset)}, Val: {len(val_asr_loader.dataset)}, Test: {len(test_asr_loader.dataset)})")
    print(f"ASR Audio Duration:      {asr_stats['total_audio_hours']*60:.2f} mins ({asr_stats['total_audio_hours']:.3f} hrs across {asr_stats['num_speakers']} speakers)")
    print(f"ASR Vocabulary:          {asr_tokenizer.vocab_size} distinct Devanagari tokens (including CTC <blank>, <pad>, <unk>, |)")
    print(f"NMT Total Pairs:         {nmt_stats['total_unique_pairs']} (Train: {len(train_nmt_loader.dataset)}, Val: {len(val_nmt_loader.dataset)}, Test: {len(test_nmt_loader.dataset)})")
    print(f"NMT Source Vocab (NE):   {nmt_stats['src_vocab_size']} tokens")
    print(f"NMT Target Vocab (EN):   {nmt_stats['tgt_vocab_size']} tokens")
    print("="*70)
    print("All datasets, manifests, vocabularies, and loaders successfully verified!\n")

if __name__ == "__main__":
    run_pipeline()
