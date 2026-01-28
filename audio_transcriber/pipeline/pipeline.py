import json
import logging
from pathlib import Path

from audio_transcriber.alignment.aligner import align_transcript
from audio_transcriber.diarization.diarizer import Diarizer
from audio_transcriber.preprocessing.audio_preprocessor import preprocess_audio
from audio_transcriber.transcription.transcriber import Transcriber
from audio_transcriber.utils.file_util import (
    load_audio_files,
    save_file,
    save_transcript_as_text,
)

logger = logging.getLogger()


class TranscriptionPipeline:
    def __init__(self, input_dir: Path, offline: bool = False):
        self.input_dir = input_dir
        self.transcriber = Transcriber(offline=offline)
        self.diarizer = Diarizer(offline=offline)

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
            if not transcribed_json_str:
                continue

            transcription_data = json.loads(transcribed_json_str)
            transcription_segments = transcription_data.get("transcription", [])

            # Diarize
            diarization_result = self.diarizer.diarize(processed_audio_file)
            if diarization_result is None:
                logging.warning(
                    f"Diarization failed for {audio_file}. "
                    "Make sure HF_TOKEN is set and you have access to the model."
                )

            # Align
            aligned_segments = align_transcript(
                transcription_segments, diarization_result
            )

            # Save
            final_data = {"transcription": aligned_segments}
            save_transcript_as_text(
                self.input_dir, transcript_file, json.dumps(final_data)
            )
            logging.info(f"Transcript {transcript_file} saved for {audio_file}")
