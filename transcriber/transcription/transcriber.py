import json
import logging
import time
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download

from transcriber.utils.constants import AI_MODEL_CONFIG
from transcriber.utils.constants import AI_MODEL_PATH
from transcriber.utils.constants import AI_MODEL_REPO
from transcriber.utils.constants import DEFAULT_LANGUAGE
from transcriber.utils.device_util import select_device_and_compute_type
from transcriber.utils.time_util import format_timestamp

logger = logging.getLogger(__name__)

class Transcriber():

    model_repo: str = AI_MODEL_REPO
    model_path: Path = AI_MODEL_PATH / model_repo

    def __init__(self):

        self.model: Optional[WhisperModel] = None

        start_load = time.time()
        device, compute_type = select_device_and_compute_type()
        logger.info(
            f"Using model path: {self.model_path} on device: {device} with compute_type: {compute_type}"
        )

        # Check if model exists locally
        if not (self.model_path / AI_MODEL_CONFIG).is_file():
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

        try:
            model = WhisperModel(
                str(self.model_path),
                device=device,
                compute_type=compute_type,
            )
            self.model = model
        except Exception as e:
            logging.error(f"Failed to load model from {self.model_path}: {e}")

        logging.info(f"Model from '{self.model_path}' loaded in {time.time() - start_load:.2f} seconds.")


    def transcribe(self, audio_file: Path) -> Optional[str]:
        if self.model is None:
            logging.error("Model is not loaded")
            return None

        logging.info(f"Transcribing: {audio_file}")
        start_transcribe = time.time()

        try:
            segments, _ = self.model.transcribe(str(audio_file), language=DEFAULT_LANGUAGE)
            lines = []
            for s in segments:
                lines.append(
                    {
                        "start": format_timestamp(s.start),
                        "end": format_timestamp(s.end),
                        "text": s.text.strip(),
                    }
                )
            transcribed_json = {"transcription": lines}
        except Exception as e:
            logging.error(f"Failed to transcribe {audio_file}: {e}")
            return None

        logging.info(
            f"Transcription completed in {time.time() - start_transcribe:.2f} seconds."
        )

        return json.dumps(transcribed_json)
