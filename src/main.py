import logging
import argparse
from pathlib import Path
from src.pipeline.pipeline import TranscriptionPipeline

logger = logging.getLogger()

def main(input_folder: str):
    pipeline = TranscriptionPipeline(input_dir)
    pipeline.run()

if __name__ == "__main__":
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


    main(args.input_folder)
