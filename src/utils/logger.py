import logging
import sys
import io

def get_logger(name: str = "nepali_voice_translator", level: str = "INFO") -> logging.Logger:
    """Creates a standardized console logger with full UTF-8 encoding support."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        try:
            stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        except Exception:
            stream = sys.stdout

        ch = logging.StreamHandler(stream)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger
