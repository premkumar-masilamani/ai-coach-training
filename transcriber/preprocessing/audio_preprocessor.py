import logging
import subprocess
import time
from pathlib import Path
from threading import Event
from typing import Optional

from transcriber.preprocessing.ffmpeg_util import get_local_ffmpeg_path

logger = logging.getLogger()


def preprocess_audio(audio_file: Path, stop_event: Optional[Event] = None) -> Path:
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

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    while process.poll() is None:
        if stop_event and stop_event.is_set():
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise InterruptedError(f"Preprocessing canceled for {audio_file}")
        time.sleep(0.1)

    _, stderr = process.communicate()
    if process.returncode != 0:
        stderr_text = stderr.strip() if stderr else "Unknown error"
        raise RuntimeError(f"ffmpeg failed preprocessing {audio_file}: {stderr_text}")

    return output_file
