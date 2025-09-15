from ..io_utils.file_loader import load_audio_files
from ..io_utils.file_writer import save_transcript
from ..preprocessing.audio_preprocessor import preprocess_audio
from ..transcription.transcriber import transcribe_audio
from ..diarization.diarizer import diarize_audio
from ..alignment.aligner import align_transcript
import os
import logging

logger = logging.getLogger()


class TranscriptionPipeline:
    def __init__(self, input_folder: str):
        self.input_folder = input_folder

    def run(self):
        audio_files = load_audio_files(self.input_folder)

        for file_path in audio_files:
            filename = os.path.splitext(os.path.basename(file_path))[0]

            # Preprocess
            processed = preprocess_audio(file_path)

            # Transcribe
            transcript = transcribe_audio(processed)

            # Diarize
            segments = diarize_audio(processed)

            # Align
            final_transcript = align_transcript(transcript, segments)

            # Save
            save_transcript(self.input_folder, filename, final_transcript)
            logging.info(f"Transcript saved for {filename}")
