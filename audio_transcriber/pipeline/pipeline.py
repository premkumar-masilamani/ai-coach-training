import logging
from pathlib import Path

from audio_transcriber.preprocessing.audio_preprocessor import preprocess_audio
from audio_transcriber.transcription.transcriber import Transcriber
from audio_transcriber.utils.file_util import (
    load_audio_files,
    save_transcript_as_text,
)

logger = logging.getLogger()


class TranscriptionPipeline:
    def __init__(self, input_dir: Path, offline: bool = False):
        self.input_dir = input_dir
        self.transcriber = Transcriber(offline=offline)

    def run(self):
        pending_files = load_audio_files(self.input_dir)
        if not pending_files:
            logging.info(f"No audio files to transcribe in {self.input_dir}")
            return

        for audio_file, transcript_file in pending_files:
            # Preprocess
            processed_audio_file = preprocess_audio(audio_file)

            # Transcribe
            transcribed_json_str = self.transcriber.transcribe(processed_audio_file)

            # Save
            if transcribed_json_str:
                save_transcript_as_text(
                    self.input_dir, transcript_file, transcribed_json_str
                )
                logging.info(f"Transcript {transcript_file} saved for {audio_file}")
