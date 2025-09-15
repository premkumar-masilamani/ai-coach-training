import logging
from pathlib import Path

logger = logging.getLogger()


def preprocess_audio(audio_file: Path) -> Path:
    logger.info(f"{audio_file} - resample, normalize, split if needed")
    return audio_file
