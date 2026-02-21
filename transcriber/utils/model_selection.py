from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from transcriber.utils.constants import AI_MODEL_WHISPER_CPP_PATH
from transcriber.utils.constants import WHISPER_CPP_MODEL_URL_PREFIX
from transcriber.utils.constants import WHISPER_CPP_REPO_MODEL_DIR


@dataclass(frozen=True)
class WhisperModelSpec:
    model_id: str
    format: str
    filename: str
    local_path: Path
    download_url: str


CONSISTENT_MODEL_ID = "medium.en"
CONSISTENT_MODEL_FORMAT = "bin"
CONSISTENT_MODEL_MIN_RAM_GB = 8


def _filename_for(model_id: str, model_format: str) -> str:
    del model_format
    return f"ggml-{model_id}.bin"


def _build_spec(model_id: str, model_format: str) -> WhisperModelSpec:
    filename = _filename_for(model_id, model_format)
    return WhisperModelSpec(
        model_id=model_id,
        format=model_format,
        filename=filename,
        local_path=AI_MODEL_WHISPER_CPP_PATH / filename,
        download_url=f"{WHISPER_CPP_MODEL_URL_PREFIX}{filename}",
    )


def min_ram_for_model(model_id: str) -> int:
    if model_id != CONSISTENT_MODEL_ID:
        raise ValueError(
            f"Unsupported Whisper model_id: {model_id}. Expected {CONSISTENT_MODEL_ID}."
        )
    return CONSISTENT_MODEL_MIN_RAM_GB


def repo_model_candidates(spec: WhisperModelSpec) -> tuple[Path, ...]:
    return (WHISPER_CPP_REPO_MODEL_DIR / spec.filename,)


def local_model_candidates(spec: WhisperModelSpec) -> tuple[Path, ...]:
    return (spec.local_path,)


def select_model_for_hardware(profile=None) -> WhisperModelSpec:
    del profile
    return _build_spec(CONSISTENT_MODEL_ID, CONSISTENT_MODEL_FORMAT)
