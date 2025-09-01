import logging
from pathlib import Path
from huggingface_hub import snapshot_download
import time
import os
from faster_whisper import WhisperModel
from typing import Optional

logger = logging.getLogger(__name__)

class TranscriptionService():

    # Model repo name
    model_repo: str = "guillaumekln/faster-whisper-medium"
    # Local path to store/load the model
    model_path: Path = Path("models") / model_repo
    # Model instance (unknown type until loaded)
    model: Optional[WhisperModel] = None

    def __init__(self):
        logging.info(f"Using model path: {self.model_path}")
        # Check if model exists locally by checking for a key file (e.g., config.json)
        if not (self.model_path / "config.json").is_file():
            logging.info("Model not found locally, starting download...")
            logging.warning(
                "This is a large model and may take a long time to download."
            )
            self.model_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Downloading from Hugging Face Hub: {self.model_repo}")
            try:
                snapshot_download(
                    repo_id=self.model_repo,
                    local_dir=str(self.model_path),
                    local_dir_use_symlinks=False,
                )
            except Exception as e:
                logging.error(f"Failed to download model: {e}")
            logging.info("Model download complete.")
        else:
            logging.info("Model found in local cache.")

        start_load = time.time()
        try:
            cpu_count = os.cpu_count() or 1
            logging.info(
                f"Loading faster-whisper model from '{self.model_path}' "
                f"with cpu_threads '{cpu_count}', and num_workers '{cpu_count}'"
            )
            model = WhisperModel(
                str(self.model_path),
                device="cpu",
                compute_type="int8",
                cpu_threads=cpu_count,
                num_workers=cpu_count,
            )
            self.model = model
        except Exception as e:
            logging.error(f"Failed to load model from {self.model_path}: {e}")

        logging.info(f"Model from '{self.model_path}' loaded in {time.time() - start_load:.2f} seconds.")


    def transcribe(self, audio_file: Path, transcript_file: Path):
        if self.model is None:
            logging.error("Model is not loaded")
            return

        logging.info(f"Transcribing: {audio_file}")
        start_transcribe = time.time()

        try:
            segments, _ = self.model.transcribe(str(audio_file))
            lines = [
                f"{self.format_timestamp(s.start)} --> {self.format_timestamp(s.end)}\n{s.text.strip()}"
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


    def format_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS,mmm format."""
        milliseconds = round(seconds * 1000)
        hours = milliseconds // 3600000
        milliseconds %= 3600000
        minutes = milliseconds // 60000
        milliseconds %= 60000
        seconds = milliseconds // 1000
        milliseconds %= 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
