import json
import logging
import os
from pathlib import Path

from audio_transcriber.utils.time_util import format_timestamp

logger = logging.getLogger()

# --- Supported audio extensions ---
audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg'}


def load_audio_files(folder_path: Path) -> list[Path]:
    # --- Find audio files to transcribe (recursively) ---
    pending_files = []
    for audio_file in folder_path.rglob("*"):
        if audio_file.suffix.lower() in audio_extensions and audio_file.is_file():
            transcript_file = audio_file.with_suffix(".json")
            if transcript_file.exists():
                logging.info(f"Skipping. Transcript already exists: {transcript_file}")
            else:
                pending_files.append((audio_file))

    return pending_files


def save_file(folder_path: Path, filename: Path, file_content: str):
    output_path = os.path.join(folder_path, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(file_content)


# TODO: Remove after the pipeline is completed
def save_transcript_as_text(folder_path: Path, filename: Path, file_content: str):
    """
    transcribed_json: JSON string like
      '{"transcription": [{"start": 0.0, "end": 8.08, "text": "..."}]}'
    """
    data = json.loads(file_content)
    segments = data.get("transcription", [])

    lines = []
    for s in segments:
        lines.append(f"{format_timestamp(s['start'])} - {format_timestamp(s['end'])} | {s["text"].strip()}")

    output_path = os.path.join(folder_path, f"{filename}_transcript.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
