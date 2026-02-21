import logging
import subprocess
import time
from pathlib import Path
from threading import Event
from typing import Optional

from transcriber.preprocessing.ffmpeg_util import get_local_ffmpeg_path

logger = logging.getLogger(__name__)
PREPROCESS_PROGRESS_LOG_INTERVAL_SECONDS = 5.0


def should_preprocess(audio_file: Path) -> bool:
    if audio_file.suffix.lower() == ".wav":
        return False

    sibling_wav = audio_file.with_suffix(".wav")
    return not sibling_wav.exists()


def preferred_wav_input(audio_file: Path) -> Path:
    if audio_file.suffix.lower() == ".wav":
        return audio_file

    sibling_wav = audio_file.with_suffix(".wav")
    if sibling_wav.exists():
        return sibling_wav
    return audio_file


def preprocessed_output_path(audio_file: Path) -> Path:
    if audio_file.suffix.lower() == ".wav" and audio_file.stem.endswith(".whisper"):
        return audio_file
    return audio_file.with_suffix("").with_suffix(".whisper.wav")


def prepare_audio_for_transcription(
        audio_file: Path, stop_event: Optional[Event] = None
) -> Path:
    source_audio = preferred_wav_input(audio_file)
    if source_audio != audio_file:
        logger.info(
            "Found existing WAV for %s. Skipping preprocessing and using %s",
            audio_file,
            source_audio,
        )

    cached_preprocessed = preprocessed_output_path(source_audio)
    if cached_preprocessed.exists():
        logger.info(
            "Found existing preprocessed audio for %s. Using %s",
            source_audio,
            cached_preprocessed,
        )
        return cached_preprocessed

    if not should_preprocess(source_audio):
        logger.info("Skipping preprocessing for WAV input: %s", source_audio)
        return source_audio

    return preprocess_audio(source_audio, stop_event=stop_event)


def preprocess_audio(audio_file: Path, stop_event: Optional[Event] = None) -> Path:
    """
    Convert any audio/video file to the Whisper-compatible format:
    16kHz, mono, 16-bit PCM WAV.
    """
    if not audio_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {audio_file}")

    ffmpeg_path = get_local_ffmpeg_path()

    output_file = preprocessed_output_path(audio_file)

    if output_file.exists():
        logger.info("Using cached preprocessed audio: %s", output_file)
        return output_file

    logger.info("Preprocessing %s -> %s", audio_file, output_file)

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
    logger.debug("ffmpeg command: %s", " ".join(command))

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    started_at = time.time()
    next_progress_log = started_at + PREPROCESS_PROGRESS_LOG_INTERVAL_SECONDS
    while process.poll() is None:
        if stop_event and stop_event.is_set():
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise InterruptedError(f"Preprocessing canceled for {audio_file}")

        now = time.time()
        if now >= next_progress_log:
            logger.info(
                "Preprocessing in progress: %s (elapsed %.0fs)",
                audio_file,
                now - started_at,
            )
            next_progress_log = now + PREPROCESS_PROGRESS_LOG_INTERVAL_SECONDS
        time.sleep(0.1)

    _, stderr = process.communicate()
    if process.returncode != 0:
        stderr_text = stderr.strip() if stderr else "Unknown error"
        raise RuntimeError(f"ffmpeg failed preprocessing {audio_file}: {stderr_text}")

    logger.info(
        "Preprocessing completed: %s (elapsed %.2fs)",
        output_file,
        time.time() - started_at,
    )
    return output_file
