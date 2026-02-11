import logging
import subprocess
from pathlib import Path

from transcriber.preprocessing.ffmpeg_util import get_local_ffmpeg_path

logger = logging.getLogger()


def preprocess_audio(audio_file: Path) -> Path:
    """
    Convert any audio/video file to the Whisper-compatible format:
    16kHz, mono, 16-bit PCM WAV.
    """
    if not audio_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {audio_file}")

    ffmpeg_path = get_local_ffmpeg_path()

    output_file = audio_file.with_suffix("").with_suffix(".whisper.wav")

    if output_file.exists():
        input_mtime = audio_file.stat().st_mtime
        output_mtime = output_file.stat().st_mtime
        if output_mtime >= input_mtime:
            logger.info(f"Using cached preprocessed audio: {output_file}")
            return output_file

    logger.info(f"Preprocessing {audio_file} -> {output_file}")

    command = [
        str(ffmpeg_path),
        "-y",
        "-i",
        str(audio_file),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_file),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "Unknown error"
        raise RuntimeError(f"ffmpeg failed preprocessing {audio_file}: {stderr}") from exc

    return output_file
