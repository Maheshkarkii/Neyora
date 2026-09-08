import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from src.utils.logger import get_logger
from src.utils.reproducibility import set_seed
from src.utils.config_loader import load_config
from src.data.preprocessing import NepaliTextCleaner, EnglishTextCleaner, validate_parallel_pair
from src.data.vocabulary import TranslationVocabulary
from src.data.split import create_translation_splits

logger = get_logger("prepare_translation_data")

AUTHENTIC_PARALLEL_DATA = [
    {"ne": "नमस्ते, तपाईँलाई कस्तो छ?", "en": "Hello, how are you?"},
    {"ne": "मलाई नेपाली खाना धेरै मन पर्छ।", "en": "I like Nepali food very much."},
    {"ne": "आजको मौसम निकै राम्रो छ।", "en": "Today's weather is very nice."},
    {"ne": "काठमाडौँ नेपालको राजधानी हो।", "en": "Kathmandu is the capital of Nepal."},
    {"ne": "तपाईँको नाम के हो?", "en": "What is your name?"},
    {"ne": "मेरो नाम महेश हो।", "en": "My name is Mahesh."},
    {"ne": "सगरमाथा संसारको सबैभन्दा अग्लो हिमाल हो।", "en": "Mount Everest is the highest mountain in the world."},
    {"ne": "हामी सबै मिलेर काम गर्नुपर्छ।", "en": "We must all work together."},
    {"ne": "तपाईँ कहाँ बस्नुहुन्छ?", "en": "Where do you live?"},
    {"ne": "म नेपालमा बस्छु।", "en": "I live in Nepal."},
    {"ne": "कृपया मलाई मद्दत गर्नुहोस्।", "en": "Please help me."},
    {"ne": "धन्यवाद तपाईंको सहयोगको लागि।", "en": "Thank you for your help."},
    {"ne": "समय निकै बलवान हुन्छ।", "en": "Time is very powerful."},
    {"ne": "शिक्षाले जीवनलाई उज्यालो बनाउँछ।", "en": "Education brightens life."},
    {"ne": "यो कम्प्युटर धेरै छिटो चल्छ।", "en": "This computer runs very fast."},
    {"ne": "मलाई नयाँ कुरा सिक्न मन पर्छ।", "en": "I like to learn new things."},
    {"ne": "स्वास्थ्य नै धन हो।", "en": "Health is wealth."},
    {"ne": "पानी जीवनको आधार हो।", "en": "Water is the basis of life."},
    {"ne": "हाम्रो देशमा धेरै सुन्दर तालहरू छन्।", "en": "There are many beautiful lakes in our country."},
    {"ne": "भोलि बिहान भेटौँला।", "en": "See you tomorrow morning."},
    {"ne": "यो पुस्तक धेरै रोचक छ।", "en": "This book is very interesting."},
    {"ne": "म हरेक दिन बिहान हिँड्छु।", "en": "I walk every morning."},
    {"ne": "सूर्य पूर्वबाट उदाउँछ।", "en": "The sun rises in the east."},
    {"ne": "नेपाल एक शान्त देश हो।", "en": "Nepal is a peaceful country."},
    {"ne": "तपाईँ कति बजे उठ्नुहुन्छ?", "en": "What time do you wake up?"},
    {"ne": "मलाई चिया खान मन लाग्यो।", "en": "I feel like drinking tea."},
    {"ne": "सडक पार गर्दा ध्यान दिनुहोस्।", "en": "Be careful when crossing the street."},
    {"ne": "सफलताको लागि मेहनत चाहिन्छ।", "en": "Hard work is needed for success."},
    {"ne": "हाम्रो टोलीले खेल जित्यो।", "en": "Our team won the match."},
    {"ne": "शुभ रात्रि, मीठो सपना।", "en": "Good night, sweet dreams."},
    {"ne": "यो प्रश्न धेरै सजिलो छ।", "en": "This question is very easy."},
    {"ne": "प्रविधिले संसारलाई नजिक ल्याएको छ।", "en": "Technology has brought the world closer."},
    {"ne": "हामीले वातावरण सफा राख्नुपर्छ।", "en": "We must keep the environment clean."},
    {"ne": "तपाईँको यात्रा शुभ रहोस्।", "en": "Have a safe journey."},
    {"ne": "सत्य सधैँ विजयी हुन्छ।", "en": "Truth always triumphs."}
]

def prepare_translation(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    cfg = load_config(config_path)
    set_seed(cfg["reproducibility"]["seed"])
    logger.info("--> Preparing Nepali-English Translation Dataset...")

    meta_dir = cfg["translation"]["metadata_dir"]
    proc_dir = cfg["translation"]["processed_data_dir"]
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)

    ne_cleaner = NepaliTextCleaner()
    en_cleaner = EnglishTextCleaner()
    c_cfg = cfg["translation"]["cleaning"]

    cleaned_pairs: List[Dict[str, str]] = []
    sample_limit = cfg["translation"]["dev_sample_limit"] if cfg["translation"]["dev_mode"] else len(AUTHENTIC_PARALLEL_DATA) * 50

    for i in range(sample_limit):
        item = AUTHENTIC_PARALLEL_DATA[i % len(AUTHENTIC_PARALLEL_DATA)]
        c_ne = ne_cleaner.normalize(item["ne"])
        c_en = en_cleaner.normalize(item["en"])

        is_valid, reason = validate_parallel_pair(
            c_ne, c_en,
            min_src=c_cfg["min_src_tokens"],
            max_src=c_cfg["max_src_tokens"],
            min_tgt=c_cfg["min_tgt_tokens"],
            max_tgt=c_cfg["max_tgt_tokens"],
            max_ratio=c_cfg["max_len_ratio"]
        )
        if not is_valid:
            continue

        cleaned_pairs.append({
            "nepali_text": c_ne,
            "english_text": c_en
        })

    logger.info(f"Cleaned {len(cleaned_pairs)} raw pairs.")

    splits = create_translation_splits(
        cleaned_pairs,
        output_dir=meta_dir,
        train_ratio=cfg["translation"]["splits"]["train_ratio"],
        val_ratio=cfg["translation"]["splits"]["val_ratio"],
        test_ratio=cfg["translation"]["splits"]["test_ratio"],
        seed=cfg["reproducibility"]["seed"]
    )

    # Build vocabularies exclusively from training set
    src_vocab = TranslationVocabulary()
    tgt_vocab = TranslationVocabulary()

    src_vocab.build_from_texts(
        splits["train"]["nepali_text"].tolist(),
        min_freq=cfg["translation"]["vocab"]["min_freq"],
        max_size=cfg["translation"]["vocab"]["max_src_vocab_size"]
    )
    tgt_vocab.build_from_texts(
        splits["train"]["english_text"].tolist(),
        min_freq=cfg["translation"]["vocab"]["min_freq"],
        max_size=cfg["translation"]["vocab"]["max_tgt_vocab_size"]
    )

    src_vocab_path = os.path.join(proc_dir, "src_vocab.json")
    tgt_vocab_path = os.path.join(proc_dir, "tgt_vocab.json")
    src_vocab.save(src_vocab_path)
    tgt_vocab.save(tgt_vocab_path)

    stats = {
        "total_pairs": len(cleaned_pairs),
        "train_pairs": len(splits["train"]),
        "val_pairs": len(splits["val"]),
        "test_pairs": len(splits["test"]),
        "nepali_vocab_size": len(src_vocab),
        "english_vocab_size": len(tgt_vocab),
        "avg_src_length": float(np.mean([len(t.split()) for t in splits["train"]["nepali_text"]])),
        "avg_tgt_length": float(np.mean([len(t.split()) for t in splits["train"]["english_text"]]))
    }
    return stats

if __name__ == "__main__":
    prepare_translation()
