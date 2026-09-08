# 🇳🇵 Nepali Voice Translator

An end-to-end deep learning system built in PyTorch that accepts Nepali speech, transcribes it to written Nepali text, and translates it into English.

## Project Structure

```text
Nep-Eng/
├── configs/
│   ├── asr_config.yaml             # ASR audio parameters, feature dims, and dataloader configs
│   ├── nmt_config.yaml             # NMT vocabulary parameters, length limits, and batch sizes
│   └── app_config.yaml             # Global application logging and seed settings
├── data/
│   ├── raw/
│   │   ├── asr/audio/              # 16kHz audio waveforms
│   │   └── nmt/                    # Raw parallel corpora
│   └── processed/
│       ├── asr/                    # train_manifest.json, val_manifest.json, test_manifest.json, vocab.json
│       └── nmt/                    # train.json, val.json, test.json, src_vocab.json, tgt_vocab.json
├── src/
│   ├── common/
│   │   ├── logger.py               # Standardized logging utility
│   │   └── utils.py                # Config management, seeding, and JSON manifest helpers
│   ├── asr/
│   │   ├── text_cleaner.py         # Devanagari Unicode NFC normalizer & symbol filter
│   │   ├── tokenizer.py            # CTC Character Tokenizer (<blank>, <pad>, <unk>, |)
│   │   ├── audio_transforms.py     # Log-Mel Spectrogram & SpecAugment extraction
│   │   ├── download_openslr54.py   # OpenSLR SLR54 data processor & speaker-disjoint splitting
│   │   └── dataset.py              # NepaliASRDataset & dynamic ASRCollateFn
│   └── nmt/
│       ├── text_cleaner.py         # Bilingual text cleaner & pair validator
│       ├── tokenizer.py            # Word/Subword Tokenizer with special tokens
│       ├── download_parallel_data.py # Parallel corpus loader, deduplicator & disjoint splitter
│       └── dataset.py              # TranslationDataset & dynamic TranslationCollateFn
├── scripts/
│   └── run_data_pipeline.py        # Automated pipeline execution & inspection script
├── tests/
│   ├── test_asr_data.py            # Unit tests for ASR transforms, cleaner, and collate
│   └── test_nmt_data.py            # Unit tests for NMT cleaner, tokenizer, and collate
└── requirements.txt
```

## Running the Data Pipeline & Tests

### Run Unit Tests
```bash
pytest -v
```

### Run the Data Pipeline
```bash
python scripts/run_data_pipeline.py
```
