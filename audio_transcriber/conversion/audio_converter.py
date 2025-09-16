import logging
from pathlib import Path

logger = logging.getLogger()

class AudioConverter:
    def __init__(self):
        pass

    def convert(self, audio_file: Path) -> Path:
        # TODO: Implement preprocessing logic
        logger.info(f"Simply returning the audio_file {audio_file}, without conversion")
        return audio_file
