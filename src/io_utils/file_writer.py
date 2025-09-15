import os
import logging

logger = logging.getLogger()


def save_transcript(folder_path: str, filename: str, transcript: str):
    output_path = os.path.join(folder_path, f"{filename}_transcript.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    return output_path
