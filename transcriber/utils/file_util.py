import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# --- Supported audio/video extensions ---
supported_file_extensions = {
    # Audio
    ".aac",
    ".ac3",
    ".aif",
    ".aifc",
    ".aiff",
    ".alac",
    ".amr",
    ".ape",
    ".au",
    ".caf",
    ".dts",
    ".eac3",
    ".flac",
    ".gsm",
    ".m4a",
    ".m4b",
    ".m4p",
    ".mid",
    ".midi",
    ".mod",
    ".mp2",
    ".mp3",
    ".mpa",
    ".mpc",
    ".oga",
    ".ogg",
    ".opus",
    ".ra",
    ".ram",
    ".s3m",
    ".spx",
    ".tak",
    ".tta",
    ".voc",
    ".wav",
    ".weba",
    ".wma",
    ".wv",
    ".xm",
    # Video / containers with audio tracks
    ".3g2",
    ".3gp",
    ".asf",
    ".avi",
    ".divx",
    ".f4v",
    ".flv",
    ".m2ts",
    ".m2v",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpe",
    ".mpeg",
    ".mpg",
    ".mts",
    ".mxf",
    ".ogm",
    ".ogv",
    ".qt",
    ".rm",
    ".rmvb",
    ".ts",
    ".vob",
    ".webm",
    ".wmv",
}


def is_preprocessed_whisper_audio(file: Path) -> bool:
    return file.suffix.lower() == ".wav" and file.stem.endswith(".whisper")


def has_original_pair_for_preprocessed(file: Path) -> bool:
    if not is_preprocessed_whisper_audio(file):
        return False

    base_stem = file.stem[: -len(".whisper")]
    for ext in supported_file_extensions:
        candidate = file.with_name(f"{base_stem}{ext}")
        if candidate != file and candidate.exists():
            return True
    return False


def transcript_path_for_audio(file: Path) -> Path:
    if is_preprocessed_whisper_audio(file):
        base_stem = file.stem[: -len(".whisper")]
        return file.with_name(f"{base_stem}.transcript.txt")
    return file.with_suffix(".transcript.txt")


def load_audio_files(folder_path: Path):
    # --- Find audio files to transcribe (recursively) ---
    pending_files = []
    seen_transcripts: set[Path] = set()
    for file in folder_path.rglob("*"):
        if file.suffix.lower() in supported_file_extensions and file.is_file():
            if has_original_pair_for_preprocessed(file):
                logger.debug(
                    "Skipping paired preprocessed file: %s (original exists)",
                    file,
                )
                continue

            transcript_file = transcript_path_for_audio(file)
            if transcript_file in seen_transcripts:
                logger.debug("Skipping duplicate transcript target for %s", file)
                continue

            if transcript_file.exists():
                logger.info("Skipping, transcript already exists: %s", transcript_file)
            else:
                pending_files.append((file, transcript_file))
                seen_transcripts.add(transcript_file)

    logger.info("Pending transcription files discovered: %s", len(pending_files))
    return pending_files


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
