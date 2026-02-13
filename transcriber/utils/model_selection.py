from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from transcriber.utils.constants import AI_MODEL_WHISPER_CPP_PATH
from transcriber.utils.constants import WHISPER_CPP_MODEL_URL_PREFIX
from transcriber.utils.constants import WHISPER_CPP_REPO_MODEL_DIR
from transcriber.utils.hardware_profile import HardwareProfile
from transcriber.utils.hardware_profile import detect_hardware_profile


@dataclass(frozen=True)
class WhisperModelSpec:
    model_id: str
    min_ram_gb: int

    @property
    def filename(self) -> str:
        return f"ggml-{self.model_id}.bin"

    @property
    def download_url(self) -> str:
        return f"{WHISPER_CPP_MODEL_URL_PREFIX}{self.filename}"

    @property
    def local_path(self) -> Path:
        return AI_MODEL_WHISPER_CPP_PATH / self.filename

    @property
    def repo_path(self) -> Path:
        return WHISPER_CPP_REPO_MODEL_DIR / self.filename


FULL_QUALITY_BASE_EN = WhisperModelSpec(model_id="base.en", min_ram_gb=8)
LOW_RAM_BASE_EN = WhisperModelSpec(model_id="base.en-q5_1", min_ram_gb=6)


def select_model_for_hardware(profile: HardwareProfile | None = None) -> WhisperModelSpec:
    profile = profile or detect_hardware_profile()

    # Processing power is the primary signal. RAM buckets only adjust
    # how aggressive we can be with full-precision models.
    threshold_by_ram_bucket = {
        "32GB+": 10,
        "16GB": 14,
        "8GB": 20,
        "<8GB": 999,
    }
    threshold = threshold_by_ram_bucket.get(profile.ram_bucket, 14)

    preferred = (
        FULL_QUALITY_BASE_EN
        if profile.processing_score >= threshold
        else LOW_RAM_BASE_EN
    )

    if profile.ram_gb < preferred.min_ram_gb:
        return LOW_RAM_BASE_EN
    return preferred


def selected_model_path(profile: HardwareProfile | None = None) -> Path:
    return select_model_for_hardware(profile).local_path
