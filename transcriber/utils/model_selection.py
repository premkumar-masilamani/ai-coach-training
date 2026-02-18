from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from transcriber.utils.constants import AI_MODEL_WHISPER_CPP_PATH
from transcriber.utils.constants import WHISPER_CPP_MODEL_URL_PREFIX
from transcriber.utils.constants import WHISPER_CPP_REPO_MODEL_DIR
from transcriber.utils.hardware_profile import HardwareProfile
from transcriber.utils.hardware_profile import detect_hardware_profile


@dataclass(frozen=True)
class ModelCatalogEntry:
    model_id: str
    min_ram_gb: int
    preferred_formats: tuple[str, ...] = ("bin",)


@dataclass(frozen=True)
class WhisperModelSpec:
    model_id: str
    format: str
    filename: str
    local_path: Path
    download_url: str


MODEL_CATALOG: tuple[ModelCatalogEntry, ...] = (
    ModelCatalogEntry(model_id="tiny.en", min_ram_gb=1),
    ModelCatalogEntry(model_id="base.en", min_ram_gb=2),
    ModelCatalogEntry(model_id="small.en", min_ram_gb=4),
    ModelCatalogEntry(model_id="medium.en", min_ram_gb=8),
    ModelCatalogEntry(model_id="large-v3", min_ram_gb=16),
)


def _filename_for(model_id: str, model_format: str) -> str:
    suffix = "bin"
    return f"ggml-{model_id}.{suffix}"


def _build_spec(model_id: str, model_format: str) -> WhisperModelSpec:
    filename = _filename_for(model_id, model_format)
    return WhisperModelSpec(
        model_id=model_id,
        format=model_format,
        filename=filename,
        local_path=AI_MODEL_WHISPER_CPP_PATH / filename,
        download_url=f"{WHISPER_CPP_MODEL_URL_PREFIX}{filename}",
    )


def _repo_path_for(model_id: str, model_format: str) -> Path:
    return WHISPER_CPP_REPO_MODEL_DIR / _filename_for(model_id, model_format)


def _find_catalog_entry(model_id: str) -> ModelCatalogEntry:
    for entry in MODEL_CATALOG:
        if entry.model_id == model_id:
            return entry
    raise ValueError(f"Unknown Whisper model_id: {model_id}")


def min_ram_for_model(model_id: str) -> int:
    return _find_catalog_entry(model_id).min_ram_gb


def format_preference_for_model(model_id: str) -> tuple[str, ...]:
    return _find_catalog_entry(model_id).preferred_formats


def repo_model_candidates(spec: WhisperModelSpec) -> tuple[Path, ...]:
    formats = format_preference_for_model(spec.model_id)
    return tuple(_repo_path_for(spec.model_id, model_format) for model_format in formats)


def local_model_candidates(spec: WhisperModelSpec) -> tuple[Path, ...]:
    formats = format_preference_for_model(spec.model_id)
    return tuple(_build_spec(spec.model_id, model_format).local_path for model_format in formats)


def _preferred_spec_for_entry(entry: ModelCatalogEntry) -> WhisperModelSpec:
    preferred_format = entry.preferred_formats[0]
    primary = _build_spec(entry.model_id, preferred_format)
    if primary.local_path.is_file():
        return primary

    for model_format in entry.preferred_formats[1:]:
        candidate = _build_spec(entry.model_id, model_format)
        if candidate.local_path.is_file():
            return candidate

    return primary


def _select_entry_by_hardware(profile: HardwareProfile) -> ModelCatalogEntry:
    fitting = [entry for entry in MODEL_CATALOG if entry.min_ram_gb <= profile.ram_gb]
    if not fitting:
        return MODEL_CATALOG[0]

    largest_min_ram = max(entry.min_ram_gb for entry in fitting)
    largest_group = [entry for entry in fitting if entry.min_ram_gb == largest_min_ram]
    if len(largest_group) == 1:
        return largest_group[0]

    # Tie-breaker reserved for future catalog expansions.
    sorted_group = sorted(largest_group, key=lambda entry: entry.model_id)
    if profile.processing_score >= 20:
        return sorted_group[-1]
    return sorted_group[0]


def select_model_for_hardware(profile: HardwareProfile | None = None) -> WhisperModelSpec:
    profile = profile or detect_hardware_profile()
    selected_entry = _select_entry_by_hardware(profile)
    return _preferred_spec_for_entry(selected_entry)


def selected_model_path(profile: HardwareProfile | None = None) -> Path:
    return select_model_for_hardware(profile).local_path
