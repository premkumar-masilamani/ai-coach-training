import logging
import argparse
import sys
from pathlib import Path

from transcriber.pipeline.pipeline import TranscriptionPipeline
from transcriber.utils.constants import DEFAULT_INPUT_DIR

logger = logging.getLogger()

def main(input_dir: Path):
    pipeline = TranscriptionPipeline(input_dir)
    pipeline.run()

if __name__ == "__main__":
    # --- Parse Arguments ---
    parser = argparse.ArgumentParser(
        description="Batch transcribe audio files using whisper.cpp."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        type=Path,
        help=f"Directory containing audio files (default: {DEFAULT_INPUT_DIR})",
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

    main(input_dir)
