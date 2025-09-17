"""
This script provides a command-line interface for batch transcribing audio files.

It takes a directory of audio files as input, processes them through a
TranscriptionPipeline, and outputs the transcriptions. The script includes
command-line arguments for specifying the input directory and enabling
verbose logging.
"""
import argparse
import logging
import sys
from pathlib import Path

from audio_transcriber.pipeline.pipeline import TranscriptionPipeline

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main(audio_files_dir: Path):
    """
    Initializes and runs the transcription pipeline for the given directory.

    Args:
        audio_files_dir: A Path object pointing to the directory containing
                         audio files to be transcribed.
    """
    try:
        logger.info(f"Starting transcription pipeline for: {audio_files_dir}")
        pipeline = TranscriptionPipeline(audio_files_dir)
        pipeline.run()
        logger.info("Transcription pipeline completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during the transcription process: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Batch transcribe audio files from a specified directory.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        type=Path,
        help="Path to the directory containing audio files to be transcribed.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level).",
    )

    args = parser.parse_args()

    # --- Logging Configuration ---
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.debug("Verbose logging enabled.")
    else:
        logger.setLevel(logging.INFO)

    input_dir: Path = args.input_dir

    # --- Input Directory Validation ---
    if not input_dir.is_dir():
        logger.error(f"Error: The specified input path is not a valid directory: {input_dir}")
        sys.exit(1)

    # --- Execute Main Function ---
    main(input_dir)
