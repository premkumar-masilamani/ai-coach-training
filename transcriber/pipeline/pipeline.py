import logging
from pathlib import Path

from transcriber.preprocessing.audio_preprocessor import preprocess_audio
from transcriber.transcription.transcriber import Transcriber
from transcriber.utils.file_util import (
    load_audio_files,
    save_file,
    save_transcript_as_text,
)

logger = logging.getLogger()


class TranscriptionPipeline:
    def __init__(self, input_dir: Path):
        self.input_dir = input_dir
        self.transcriber = Transcriber()

    def run(self):
        pending_files = load_audio_files(self.input_dir)
        if not pending_files:
            logging.info(f"No audio files to transcribe in {self.input_dir}")
            return

        for audio_file, transcript_file in pending_files:
            # Preprocess
            processed_audio_file = preprocess_audio(audio_file)

            # Transcribe
            transcribed_json = self.transcriber.transcribe(processed_audio_file)
            # save_file(self.input_dir, transcript_file, transcribed_json)

            # Save
            if transcribed_json:
                save_transcript_as_text(
                    self.input_dir, transcript_file, transcribed_json
                )
                logging.info(f"Transcript {transcript_file} saved for {audio_file}")
