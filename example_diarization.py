#!/usr/bin/env python3
"""
Example usage of the offline diarization service.

This script demonstrates how to use the PyannoteDiarizer class to perform
speaker diarization on audio files with complete offline functionality
after the initial model download.
"""

import logging
import os
import sys
from pathlib import Path
from src.diarization.diarizer import Diarizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main function to demonstrate diarization usage."""

    # Example audio file path - update this to your audio file
    audio_file = "/path/to/your/audio/file.wav"

    # Check if audio file exists
    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        print("Please update the 'audio_file' variable with a valid audio file path.")
        sys.exit(1)

    try:
        print("Initializing diarization service...")
        print("Note: On first run, this will download the model (~1GB) if not cached locally.")
        print("Subsequent runs will be completely offline.\n")

        # Initialize the diarizer
        # The HF_TOKEN can be set as an environment variable or passed directly
        # For first-time download only: export HF_TOKEN="your_hugging_face_token"
        diarizer = Diarizer()

        if diarizer.pipeline is None:
            print("Failed to initialize diarization pipeline.")
            print("Make sure you have a valid Hugging Face token for the initial download.")
            print("Set it as: export HF_TOKEN='your_token_here'")
            sys.exit(1)

        print(f"Processing audio file: {audio_file}")

        # Perform diarization
        # Optional: specify min/max speakers if known
        diarization_result = diarizer.diarize(
            audio_file,
            min_speakers=2,  # Optional: minimum number of speakers
            max_speakers=4   # Optional: maximum number of speakers
        )

        if diarization_result is None:
            print("Diarization failed. Check the logs for more details.")
            sys.exit(1)

        # Format and display results
        formatted_output = diarizer.format_diarization_output(diarization_result, audio_file)
        print("\n" + formatted_output)

        # Get speaker count
        speaker_count = diarizer.get_speaker_count(diarization_result)
        print(f"\nTotal speakers detected: {speaker_count}")

        # Save results to file
        output_file = Path(audio_file).with_suffix('.diarization.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(formatted_output)

        print(f"\nDiarization results saved to: {output_file}")

        # Example: Iterate through segments programmatically
        print("\nSegment details:")
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            duration = turn.end - turn.start
            print(f"  {speaker}: {turn.start:.2f}s - {turn.end:.2f}s ({duration:.2f}s)")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def batch_diarize_directory(input_dir: str, audio_extensions: set = None):
    """
    Example function to batch process multiple audio files in a directory.

    Args:
        input_dir (str): Directory containing audio files
        audio_extensions (set): Set of audio file extensions to process
    """
    if audio_extensions is None:
        audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg'}

    input_path = Path(input_dir)
    if not input_path.is_dir():
        print(f"Invalid directory: {input_dir}")
        return

    # Find audio files
    audio_files = []
    for file in input_path.rglob("*"):
        if file.suffix.lower() in audio_extensions and file.is_file():
            # Check if diarization file already exists
            diarization_file = file.with_suffix('.diarization.txt')
            if not diarization_file.exists():
                audio_files.append(file)
            else:
                print(f"Skipping {file.name} - diarization already exists")

    if not audio_files:
        print(f"No audio files to process in {input_dir}")
        return

    print(f"Found {len(audio_files)} audio files to process")

    # Initialize diarizer once for batch processing
    diarizer = Diarizer()

    if diarizer.pipeline is None:
        print("Failed to initialize diarization pipeline for batch processing.")
        return

    # Process each file
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\nProcessing {i}/{len(audio_files)}: {audio_file.name}")

        try:
            diarization_result = diarizer.diarize(str(audio_file))

            if diarization_result is not None:
                # Save diarization results
                output_file = audio_file.with_suffix('.diarization.txt')
                formatted_output = diarizer.format_diarization_output(
                    diarization_result, str(audio_file)
                )

                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(formatted_output)

                speaker_count = diarizer.get_speaker_count(diarization_result)
                print(f"  → Completed: {speaker_count} speakers detected")
                print(f"  → Saved to: {output_file}")
            else:
                print(f"  → Failed to process {audio_file.name}")

        except Exception as e:
            print(f"  → Error processing {audio_file.name}: {e}")

if __name__ == "__main__":
    # Check if batch processing was requested
    if len(sys.argv) > 1:
        if sys.argv[1] == "batch" and len(sys.argv) > 2:
            print("Running in batch mode...")
            batch_diarize_directory(sys.argv[2])
        else:
            print("Usage for batch processing:")
            print("  python3 example_diarization.py batch /Users/premkumar/Downloads/AudioFiles")
    else:
        # Run single file example
        main()
