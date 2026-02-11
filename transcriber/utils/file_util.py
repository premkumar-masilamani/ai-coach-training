import json
import logging
import os
from pathlib import Path

logger = logging.getLogger()

# --- Supported audio/video extensions ---
audio_extensions = {
    ".aac",
    ".aiff",
    ".alac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
    ".wmv",
}


def is_preprocessed_whisper_audio(file: Path) -> bool:
    return file.suffix.lower() == ".wav" and file.stem.endswith(".whisper")


def has_original_pair_for_preprocessed(file: Path) -> bool:
    if not is_preprocessed_whisper_audio(file):
        return False

    base_stem = file.stem[: -len(".whisper")]
    for ext in audio_extensions:
        candidate = file.with_name(f"{base_stem}{ext}")
        if candidate != file and candidate.exists():
            return True
    return False


def transcript_path_for_audio(file: Path) -> Path:
    if is_preprocessed_whisper_audio(file):
        base_stem = file.stem[: -len(".whisper")]
        return file.with_name(f"{base_stem}.txt")
    return file.with_suffix(".txt")


def load_audio_files(folder_path: Path):
    # --- Find audio files to transcribe (recursively) ---
    pending_files = []
    seen_transcripts: set[Path] = set()
    for file in folder_path.rglob("*"):
        if file.suffix.lower() in audio_extensions and file.is_file():
            if has_original_pair_for_preprocessed(file):
                logging.info(
                    "Skipping paired preprocessed file: %s (original exists)",
                    file,
                )
                continue

            transcript_file = transcript_path_for_audio(file)
            if transcript_file in seen_transcripts:
                logging.info("Skipping duplicate transcript target for %s", file)
                continue

            if transcript_file.exists():
                logging.info(f"Skipping. Transcript already exists: {transcript_file}")
            else:
                pending_files.append((file, transcript_file))
                seen_transcripts.add(transcript_file)

    return pending_files


def save_file(folder_path: Path, filename: str, file_content: str):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(file_content)


# TODO: Remove after the pipeline is completed
def save_transcript_as_text(folder_path: Path, filename: str, file_content: str):
    """
    transcribed_json: JSON string like
      '{"transcription": [{"start": 0.0, "end": 8.08, "text": "..."}]}'
    """
    data = json.loads(file_content)
    segments = data.get("transcription", [])

    lines = []
    for s in segments:
        start_val = s.get("start")
        end_val = s.get("end")
        if isinstance(start_val, (int, float)):
            start = f"{start_val:.2f}"
        else:
            start = str(start_val)
        if isinstance(end_val, (int, float)):
            end = f"{end_val:.2f}"
        else:
            end = str(end_val)
        text = s["text"].strip()
        lines.append(f"{start} - {end} | {text}")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
