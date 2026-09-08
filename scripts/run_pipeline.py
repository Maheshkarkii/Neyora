import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.utils.logger import get_logger
from src.utils.config_loader import load_config
from src.data.preprocessing import AudioPreprocessor
from src.data.vocabulary import ASRVocabulary, TranslationVocabulary
from src.data.tokenizer import ASRTokenizer, TranslationTokenizer
from src.data.collate import get_asr_dataloaders, get_translation_dataloaders
from scripts.prepare_asr_data import prepare_asr
from scripts.prepare_translation_data import prepare_translation
from scripts.inspect_data import inspect

logger = get_logger("run_pipeline")

def main():
    logger.info("==========================================================")
    logger.info("   PHASE 1 COMPLETE DATA PIPELINE EXECUTION & VERIFICATION")
    logger.info("==========================================================")

    # 1. Run Data Preparation
    asr_stats = prepare_asr("configs/config.yaml")
    trans_stats = prepare_translation("configs/config.yaml")

    cfg = load_config("configs/config.yaml")

    # 2. Test ASR DataLoaders
    logger.info("Verifying ASR PyTorch DataLoaders...")
    preprocessor = AudioPreprocessor(
        target_sample_rate=cfg["asr"]["target_sample_rate"],
        n_mels=cfg["asr"]["features"]["n_mels"],
        n_fft=cfg["asr"]["features"]["n_fft"],
        hop_length=cfg["asr"]["features"]["hop_length"],
        win_length=cfg["asr"]["features"]["win_length"]
    )
    asr_vocab = ASRVocabulary.load(os.path.join(cfg["asr"]["processed_data_dir"], "vocab.json"))
    asr_tok = ASRTokenizer(asr_vocab)

    train_asr_loader, val_asr_loader, test_asr_loader = get_asr_dataloaders(
        train_csv=os.path.join(cfg["asr"]["metadata_dir"], "asr_train.csv"),
        val_csv=os.path.join(cfg["asr"]["metadata_dir"], "asr_val.csv"),
        test_csv=os.path.join(cfg["asr"]["metadata_dir"], "asr_test.csv"),
        preprocessor=preprocessor,
        tokenizer=asr_tok,
        batch_size=cfg["asr"]["dataloader"]["batch_size"],
        num_workers=cfg["asr"]["dataloader"]["num_workers"],
        pin_memory=cfg["asr"]["dataloader"]["pin_memory"]
    )

    specs, targets, in_lens, tgt_lens, texts = next(iter(train_asr_loader))
    logger.info(f"ASR Train Batch: Mel Specs = {list(specs.shape)} [B, n_mels, time] ({specs.dtype})")
    logger.info(f"ASR Train Batch: Targets   = {list(targets.shape)} [B, max_tokens] ({targets.dtype})")
    logger.info(f"ASR Train Batch: Input Lengths = {in_lens.tolist()[:4]}...")
    logger.info(f"ASR Train Batch: Target Lengths= {tgt_lens.tolist()[:4]}...")

    # 3. Test Translation DataLoaders
    logger.info("Verifying Translation PyTorch DataLoaders...")
    src_vocab = TranslationVocabulary.load(os.path.join(cfg["translation"]["processed_data_dir"], "src_vocab.json"))
    tgt_vocab = TranslationVocabulary.load(os.path.join(cfg["translation"]["processed_data_dir"], "tgt_vocab.json"))

    src_tok = TranslationTokenizer(src_vocab, is_nepali=True)
    tgt_tok = TranslationTokenizer(tgt_vocab, is_nepali=False)

    train_nmt_loader, val_nmt_loader, test_nmt_loader = get_translation_dataloaders(
        train_csv=os.path.join(cfg["translation"]["metadata_dir"], "translation_train.csv"),
        val_csv=os.path.join(cfg["translation"]["metadata_dir"], "translation_val.csv"),
        test_csv=os.path.join(cfg["translation"]["metadata_dir"], "translation_test.csv"),
        src_tokenizer=src_tok,
        tgt_tokenizer=tgt_tok,
        batch_size=cfg["translation"]["dataloader"]["batch_size"],
        num_workers=cfg["translation"]["dataloader"]["num_workers"],
        pin_memory=cfg["translation"]["dataloader"]["pin_memory"]
    )

    src_batch, src_lens, tgt_batch, tgt_lens, raw_srcs, raw_tgts = next(iter(train_nmt_loader))
    logger.info(f"NMT Train Batch: Source    = {list(src_batch.shape)} [B, max_src_len] ({src_batch.dtype})")
    logger.info(f"NMT Train Batch: Target    = {list(tgt_batch.shape)} [B, max_tgt_len] ({tgt_batch.dtype})")
    logger.info(f"NMT Train Batch: Src Lengths = {src_lens.tolist()[:4]}...")
    logger.info(f"NMT Train Batch: Tgt Lengths = {tgt_lens.tolist()[:4]}...")

    # 4. Print Data Inspection
    inspect()

    print("="*70)
    print("                    DATASET PIPELINE STATISTICS                    ")
    print("="*70)
    print("ASR (Speech-to-Text):")
    print(f"  - Total Audio Samples: {asr_stats['total_samples']} (Train: {asr_stats['train_samples']}, Val: {asr_stats['val_samples']}, Test: {asr_stats['test_samples']})")
    print(f"  - Unique Speakers:     {asr_stats['num_speakers']} (Speaker-Disjoint Split)")
    print(f"  - Duration:            Avg={asr_stats['avg_duration']:.2f}s, Min={asr_stats['min_duration']:.2f}s, Max={asr_stats['max_duration']:.2f}s")
    print(f"  - Vocabulary:          {asr_stats['vocab_size']} Devanagari characters (incl. CTC tokens)")
    print("\nTranslation (Nepali -> English):")
    print(f"  - Total Parallel Pairs:{trans_stats['total_pairs']} (Train: {trans_stats['train_pairs']}, Val: {trans_stats['val_pairs']}, Test: {trans_stats['test_pairs']})")
    print(f"  - Nepali Vocab Size:   {trans_stats['nepali_vocab_size']} tokens")
    print(f"  - English Vocab Size:  {trans_stats['english_vocab_size']} tokens")
    print(f"  - Avg Source Length:   {trans_stats['avg_src_length']:.2f} tokens")
    print(f"  - Avg Target Length:   {trans_stats['avg_tgt_length']:.2f} tokens")
    print("="*70)
    print("Phase 1 verification completed successfully!\n")

if __name__ == "__main__":
    main()
