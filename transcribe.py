import logging
import time
import argparse
import sys
import os
from pathlib import Path
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
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


def diarize_speakers(audio_file: Path, hf_token: str):
    logging.info("Performing speaker diarization...")
    start_diarize = time.time()
    try:
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        diarization_pipeline.to(torch.device(device))
        diarization = diarization_pipeline(str(audio_file))
        logging.info(f"Diarization completed in {time.time() - start_diarize:.2f} seconds.")
        return diarization
    except Exception as e:
        logging.error(f"Failed to perform diarization: {e}")
        logging.error("Please ensure you have a valid Hugging Face token and have accepted the user agreement for the pyannote/speaker-diarization-3.1 model.")
        sys.exit(1)


class SpeakerAligner:
    def align(self, transcription_segments, diarization):
        aligned_transcriptions = []

        for segment in transcription_segments:
            for word in segment.words:
                word_start = word.start
                word_end = word.end

                best_match = self.find_best_match(diarization, word_start, word_end)
                if best_match:
                    speaker = best_match[2]
                    aligned_transcriptions.append({
                        "speaker": speaker,
                        "start": word_start,
                        "end": word_end,
                        "text": word.word
                    })
        return aligned_transcriptions

    def find_best_match(self, diarization, start_time, end_time):
        best_match = None
        max_intersection = 0

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            intersection_start = max(start_time, turn.start)
            intersection_end = min(end_time, turn.end)
            intersection_length = intersection_end - intersection_start

            if intersection_length > max_intersection:
                max_intersection = intersection_length
                best_match = (turn.start, turn.end, speaker)

        return best_match

    def merge_consecutive_segments(self, segments):
        if not segments:
            return []

        merged = []
        current_segment = segments[0]

        for i in range(1, len(segments)):
            next_segment = segments[i]
            if next_segment["speaker"] == current_segment["speaker"] and (next_segment["start"] - current_segment["end"] < 0.5):
                current_segment["end"] = next_segment["end"]
                current_segment["text"] += "" + next_segment["text"]
            else:
                merged.append(current_segment)
                current_segment = next_segment

        merged.append(current_segment)
        return merged


def transcribe_and_align(model: WhisperModel, audio_file: Path, transcript_file: Path, diarization):
    logging.info(f"Transcribing and aligning: {audio_file}")
    start_transcribe = time.time()

    try:
        segments, _ = model.transcribe(str(audio_file), word_timestamps=True)

        aligner = SpeakerAligner()
        aligned_segments = aligner.align(segments, diarization)
        merged_segments = aligner.merge_consecutive_segments(aligned_segments)

        output_lines = []
        for segment in merged_segments:
            start_time = format_timestamp(segment['start'])
            end_time = format_timestamp(segment['end'])
            speaker = segment['speaker']
            text = segment['text'].strip()
            output_lines.append(f"[{start_time} --> {end_time}] {speaker}: {text}")

        full_text = "\n\n".join(output_lines)
        
    except Exception as e:
        logging.error(f"Failed to transcribe and align {audio_file}: {e}")
        return

    logging.info(
        f"Transcription and alignment completed in {time.time() - start_transcribe:.2f} seconds."
    )

    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(full_text.strip())
        logging.info(f"Transcription saved to: {transcript_file}")


def main():
    # --- Parse Arguments ---
    parser = argparse.ArgumentParser(
        description="Batch transcribe audio files with speaker diarization."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing audio files",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="Hugging Face authentication token for pyannote.audio. Can also be set via HUGGING_FACE_TOKEN environment variable.",
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

    # --- Get Hugging Face Token ---
    hf_token = args.hf_token or os.environ.get("HUGGING_FACE_TOKEN")
    if not hf_token:
        logging.error("Hugging Face token not provided. Please pass it using the --hf-token argument or set the HUGGING_FACE_TOKEN environment variable.")
        sys.exit(1)

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

    # --- Load Models and Run Transcription ---
    transcription_model = load_transcription_model()

    for audio_file, transcript_file in pending_files:
        diarization = diarize_speakers(audio_file, hf_token)
        transcribe_and_align(transcription_model, audio_file, transcript_file, diarization)


if __name__ == "__main__":
    main()
