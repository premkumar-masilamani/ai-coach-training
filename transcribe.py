import logging
import time
import argparse
import sys
import os
from pathlib import Path
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download

# --- Configure Root Logger ---
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS,mmm format."""
    milliseconds = round(seconds * 1000)
    hours = milliseconds // 3600000
    milliseconds %= 3600000
    minutes = milliseconds // 60000
    milliseconds %= 60000
    seconds = milliseconds // 1000
    milliseconds %= 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def load_transcription_model():
    # Define model path structure
    model_repo = "guillaumekln/faster-whisper-medium"
    model_path = Path("models") / model_repo

    logging.info(f"Using model path: {model_path}")

    # Check if model exists locally by checking for a key file (e.g., config.json)
    if not (model_path / "config.json").is_file():
        logging.info("Model not found locally, starting download...")
        logging.warning(
            "This is a large model and may take a long time to download."
        )
        model_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Downloading from Hugging Face Hub: {model_repo}")
        try:
            snapshot_download(
                repo_id=model_repo,
                local_dir=str(model_path),
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            logging.error(f"Failed to download model: {e}")
            sys.exit(1)
        logging.info("Model download complete.")
    else:
        logging.info("Model found in local cache.")

    start_load = time.time()
    try:
        cpu_count = os.cpu_count() or 1
        logging.info(
            f"Loading faster-whisper model from '{model_path}' "
            f"with cpu_threads '{cpu_count}', and num_workers '{cpu_count}'"
        )
        model = WhisperModel(
            str(model_path),
            cpu_threads=cpu_count,
            num_workers=cpu_count,
        )
    except Exception as e:
        logging.error(f"Failed to load model from {model_path}: {e}")
        sys.exit(1)

    logging.info(f"Model from '{model_path}' loaded in {time.time() - start_load:.2f} seconds.")
    return model


def transcribe(model: WhisperModel, audio_file: Path, transcript_file: Path):
    logging.info(f"Transcribing: {audio_file}")
    start_transcribe = time.time()

    try:
        segments, _ = model.transcribe(str(audio_file))
        lines = [
            f"{format_timestamp(s.start)} --> {format_timestamp(s.end)}\n{s.text.strip()}"
            for s in segments
        ]
        full_text = "\n\n".join(lines)
    except Exception as e:
        logging.error(f"Failed to transcribe {audio_file}: {e}")
        return

    logging.info(
        f"Transcription completed in {time.time() - start_transcribe:.2f} seconds."
    )

    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(full_text.strip())
        logging.info(f"Transcription saved to: {transcript_file}")


def main():
    # --- Parse Arguments ---
    parser = argparse.ArgumentParser(
        description="Batch transcribe audio files using faster-whisper."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing audio files",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG level) logging.",
    )

    args = parser.parse_args()

    # --- Configure Logging Level ---
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.debug("Verbose logging enabled.")
    else:
        logger.setLevel(logging.INFO)

    input_dir: Path = args.input_dir

    # --- Validate input directory ---
    if not input_dir.is_dir():
        logging.error(f"Invalid directory: {input_dir}")
        sys.exit(1)

    # --- Supported audio extensions ---
    audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg'}

    # --- Find audio files to transcribe (recursively) ---
    pending_files = []
    for file in input_dir.rglob("*"):
        if file.suffix.lower() in audio_extensions and file.is_file():
            transcript_file = file.with_suffix(".txt")
            if transcript_file.exists():
                logging.info(f"Skipping. Transcript already exists: {transcript_file}")
            else:
                pending_files.append((file, transcript_file))

    if not pending_files:
        logging.info(f"No audio files to transcribe in {input_dir}")
        sys.exit(0)

    # --- Load and Run Transcriber ---
    model = load_transcription_model()
    for audio_file, transcript_file in pending_files:
        transcribe(model, audio_file, transcript_file)


if __name__ == "__main__":
    main()
