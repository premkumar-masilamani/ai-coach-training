import json
import logging
from pathlib import Path
from huggingface_hub import snapshot_download
import time
from faster_whisper import WhisperModel
from typing import Optional

logger = logging.getLogger(__name__)

AI_MODEL_PATH = Path("ai_models")
STR_DEVICE_CPU = "cpu"
STR_COMPUTE_TYPE_FLOAT32 = "float32"

class Transcriber():

    model_repo: str = "guillaumekln/faster-whisper-medium"
    model_path: Path = AI_MODEL_PATH / model_repo

    def __init__(self):

        self.model: Optional[WhisperModel] = None

        start_load = time.time()
        logger.info(f"Using model path: {self.model_path} on device: {STR_DEVICE_CPU} with compute_type: {STR_COMPUTE_TYPE_FLOAT32}")

        # Check if model exists locally
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

        try:
            model = WhisperModel(
                str(self.model_path),
                device=STR_DEVICE_CPU,
                compute_type=STR_COMPUTE_TYPE_FLOAT32
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
            segments, _ = self.model.transcribe(str(audio_file))
            lines = []
            for s in segments:
                lines.append({
                    "start": s.start,
                    "end": s.end,
                    "text": s.text.strip(),
                })
            transcribed_json = {"transcription": lines}
        except Exception as e:
            logging.error(f"Failed to transcribe {audio_file}: {e}")
            return None

        logging.info(
            f"Transcription completed in {time.time() - start_transcribe:.2f} seconds."
        )

        return json.dumps(transcribed_json)
