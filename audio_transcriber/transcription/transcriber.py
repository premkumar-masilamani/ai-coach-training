import json
import logging
import time
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download

from audio_transcriber.utils.constants import AI_MODEL_PATH, STR_DEVICE_CPU, STR_COMPUTE_TYPE_FLOAT32

logger = logging.getLogger(__name__)


class Transcriber():
    """
    A class to handle audio transcription using the faster-whisper library.

    This class manages the download, loading, and execution of a speech-to-text
    model from Hugging Face Hub. It is designed to transcribe audio files into
    structured JSON output.
    """

    model_repo: str = "guillaumekln/faster-whisper-medium"
    """The repository ID of the model on Hugging Face Hub."""

    model_path: Path = AI_MODEL_PATH / model_repo
    """The local path to store the downloaded model."""

    def __init__(self):
        """
        Initializes the Transcriber, including downloading and loading the model.

        This constructor checks if the transcription model is available locally.
        If not, it downloads the model from the Hugging Face Hub. It then loads
        the model into memory for transcription tasks.
        """
        logger.info("Initializing Transcriber...")
        self.model: Optional[WhisperModel] = None

        start_load = time.time()
        logger.info(f"Transcriber initialization started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(
            f"Using model path: {self.model_path} on device: {STR_DEVICE_CPU} with compute_type: {STR_COMPUTE_TYPE_FLOAT32}")

        # Check if model exists locally
        if not (self.model_path / "config.json").is_file():
            logger.info("Model configuration file not found locally, starting download...")
            logging.warning(
                "This is a large model and may take a long time to download."
            )
            self.model_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading model from Hugging Face Hub: {self.model_repo} to {self.model_path}")
            try:
                snapshot_download(
                    repo_id=self.model_repo,
                    local_dir=str(self.model_path),
                    local_dir_use_symlinks=False,
                )
                logger.info("Model download complete.")
            except Exception as e:
                logger.error(f"Failed to download model: {e}", exc_info=True)
                raise
        else:
            logger.info("Model found in local cache.")

        try:
            logger.info(f"Loading model from {self.model_path}...")
            self.model = WhisperModel(
                str(self.model_path),
                device=DEFAULT_DEVICE_CPU,
                compute_type=DEFAULT_COMPUTE_TYPE
            )
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}", exc_info=True)
            raise

        logging.info(f"Model initialization finished in {time.time() - start_load:.2f} seconds.")

    def transcribe(self, audio_file: Path) -> Optional[str]:
        """
        Transcribes an audio file into a JSON string with transcription segments.

        Args:
            audio_file: The path to the audio file to be transcribed.

        Returns:
            A JSON string containing the transcription, with each segment having
            'start' and 'end' times and the transcribed 'text'. Returns None if
            the model is not loaded or if transcription fails.
        """
        if self.model is None:
            logger.error("Cannot transcribe: Model is not loaded.")
            return None

        logger.info(f"Starting transcription for: {audio_file}")
        start_transcribe = time.time()

        try:
            logger.info(f"Beginning transcription process for {audio_file} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.debug(f"Calling model.transcribe for {audio_file}")
            segments, info = self.model.transcribe(str(audio_file), word_timestamps=True)
            logger.info(f"Detected language '{info.language}' with {info.language_probability:.2f} probability.")

            logger.info("This may take a while, transcribing audio...")
            segments = list(segments)
            logger.info(f"Transcription complete. Got {len(segments)} segments.")

            lines = []
            logger.info("Processing transcribed segments...")
            for s in segments:
                line = {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text.strip(),
                }
                lines.append(line)
                logger.debug(f"Processed segment from {s.start:.2f}s to {s.end:.2f}s: {s.text.strip()}")

            transcribed_json = {"transcription": lines}
            logger.info(f"Successfully transcribed and processed {len(lines)} segments.")
        except Exception as e:
            logger.error(f"Failed to transcribe {audio_file}: {e}", exc_info=True)
            return None

        logger.info(
            f"Transcription of {audio_file} completed in {time.time() - start_transcribe:.2f} seconds."
        )

        return json.dumps(transcribed_json)
