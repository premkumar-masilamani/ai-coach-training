import logging
from pyannote.core import Segment

logger = logging.getLogger(__name__)


def align_transcript(transcription_segments: list, diarization_result) -> list:
    """
    Aligns transcription segments with diarization results to assign speakers.

    Args:
        transcription_segments (list): List of dicts with 'start', 'end', and 'text'.
        diarization_result (pyannote.core.Annotation): Diarization result.

    Returns:
        list: List of dicts with 'start', 'end', 'text', and 'speaker'.
    """
    if diarization_result is None:
        logger.warning("No diarization result provided.")

    aligned_segments = []
    for segment in transcription_segments:
        start = segment["start"]
        end = segment["end"]

        speaker = "UNKNOWN"

        if diarization_result is not None:
            # Find speaker with max overlap
            query_segment = Segment(start, end)
            overlap = diarization_result.crop(query_segment)

            if overlap:
                speaker_durations = {}
                for sub_segment, _, spk in overlap.itertracks(yield_label=True):
                    speaker_durations[spk] = (
                        speaker_durations.get(spk, 0) + sub_segment.duration
                    )

                if speaker_durations:
                    speaker = max(speaker_durations, key=speaker_durations.get)

        aligned_segments.append({**segment, "speaker": speaker})

    return aligned_segments
