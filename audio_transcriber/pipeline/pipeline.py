import logging
from pathlib import Path

from audio_transcriber.alignment.aligner import Aligner
from audio_transcriber.conversion.audio_converter import convert_audio
from audio_transcriber.diarization.diarizer import Diarizer
from audio_transcriber.transcription.transcriber import Transcriber
from audio_transcriber.utils.file_util import get_audio_files_list, save_file, save_transcript_as_text

logger = logging.getLogger()


class TranscriptionPipeline:
    def __init__(self, input_dir: Path):
        self.input_dir = input_dir
        self.transcriber = Transcriber()
        self.diarizer = Diarizer()
        self.aligner = Aligner()

    def run(self):
        audio_files_list = get_audio_files_list(self.input_dir)
        if not audio_files_list:
            logging.info(f"No audio files to transcribe in {self.input_dir}")
            return

        for audio_file in audio_files_list:

            # TODO: Audio Conversion
            converted_audio_file = convert_audio(audio_file)

            # Transcribe
            transcribed_json = self.transcriber.transcribe(converted_audio_file)
            # TODO: Move to Save Section
            if transcribed_json:
                transcript_txt_filename = audio_file.with_name(audio_file.stem + "_transcript.txt")
                save_transcript_as_text(self.input_dir, transcript_txt_filename, transcribed_json)

                transcript_json_filename = audio_file.with_name(audio_file.stem + "_transcript.json")
                save_file(self.input_dir, transcript_json_filename, transcribed_json)
                logging.info(f"Transcript {transcript_json_filename} saved for {audio_file}")

            # Diarize
            diarized_json = self.diarizer.diarize(converted_audio_file)
            # TODO: Move to Save Section
            if diarized_json:
                diarized_json_filename = audio_file.with_name(audio_file.stem + "_diarized.json")
                save_file(self.input_dir, diarized_json_filename, diarized_json)
                logging.info(f"Diarization {diarized_json_filename} saved for {audio_file}")

            # TODO: Align
            # final_transcript = self.aligner.align(transcribed_json, diarized_json)

            # Save
            # TODO: Move all the file saving logics in this step.

            # TODO: Clean up the temporary files
