import logging
import argparse
import sys
from pathlib import Path
from services.transcription_service import TranscriptionService

logger = logging.getLogger()

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
    model = TranscriptionService()
    for audio_file, transcript_file in pending_files:
        model.transcribe(audio_file, transcript_file)


if __name__ == "__main__":
    main()
