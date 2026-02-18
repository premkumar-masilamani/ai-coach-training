import logging
from pathlib import Path

from transcriber.preprocessing.audio_preprocessor import (
    prepare_audio_for_transcription,
)
from transcriber.transcription.transcriber import Transcriber
from transcriber.utils.file_util import (
    load_audio_files,
    save_transcript_as_text,
)

logger = logging.getLogger(__name__)


class TranscriptionPipeline:
    def __init__(self, input_dir: Path):
        self.input_dir = input_dir
        self.transcriber = Transcriber()

    def run(self):
        logger.info("Scanning input directory: %s", self.input_dir)
        pending_files = load_audio_files(self.input_dir)
        if not pending_files:
            logger.info("No audio files to transcribe in %s", self.input_dir)
            return

        total = len(pending_files)
        for index, (audio_file, transcript_file) in enumerate(pending_files, start=1):
            logger.info("Processing file %s/%s: %s", index, total, audio_file)
            processed_audio_file = prepare_audio_for_transcription(audio_file)

            # Transcribe
            transcribed_json = self.transcriber.transcribe(processed_audio_file)

            # Save
            if transcribed_json:
                save_transcript_as_text(
                    self.input_dir, transcript_file, transcribed_json
                )
                logger.info("Transcript saved: %s", transcript_file)
            else:
                logger.warning("Transcription failed for %s", audio_file)
