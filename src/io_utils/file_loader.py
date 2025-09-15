import os
import logging

logger = logging.getLogger()


def load_audio_files(folder_path: str):
    audio_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(('.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg'))
    ]
    return audio_files
