import unittest
from pyannote.core import Annotation, Segment
from audio_transcriber.alignment.aligner import align_transcript

class TestAligner(unittest.TestCase):
    def test_align_transcript(self):
        # Mock transcription segments
        transcription_segments = [
            {"start": 0.0, "end": 2.0, "text": "Hello world"},
            {"start": 2.5, "end": 4.5, "text": "This is a test"},
            {"start": 5.0, "end": 7.0, "text": "Goodbye"}
        ]

        # Mock diarization result
        diarization = Annotation()
        diarization[Segment(0.0, 3.0)] = "SPEAKER_01"
        diarization[Segment(3.0, 8.0)] = "SPEAKER_02"

        aligned = align_transcript(transcription_segments, diarization)

        self.assertEqual(len(aligned), 3)

        # Segment 0: 0.0-2.0, should be SPEAKER_01
        self.assertEqual(aligned[0]["speaker"], "SPEAKER_01")

        # Segment 1: 2.5-4.5
        # Overlap with SPEAKER_01: 2.5-3.0 (0.5s)
        # Overlap with SPEAKER_02: 3.0-4.5 (1.5s)
        # Should be SPEAKER_02
        self.assertEqual(aligned[1]["speaker"], "SPEAKER_02")

        # Segment 2: 5.0-7.0, should be SPEAKER_02
        self.assertEqual(aligned[2]["speaker"], "SPEAKER_02")

    def test_align_transcript_no_diarization(self):
        transcription_segments = [
            {"start": 0.0, "end": 2.0, "text": "Hello world"}
        ]
        aligned = align_transcript(transcription_segments, None)
        self.assertEqual(aligned[0]["speaker"], "UNKNOWN")

if __name__ == "__main__":
    unittest.main()
