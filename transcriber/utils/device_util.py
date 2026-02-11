import logging
import platform
from typing import Tuple

logger = logging.getLogger(__name__)


def _pick_compute_type(device: str, supported: set[str]) -> str:
    if device == "cuda":
        for candidate in ("float16", "int8_float16", "int8", "float32", "int8_float32"):
            if candidate in supported:
                return candidate
    else:
        for candidate in ("int8", "int8_float32", "float32", "int16"):
            if candidate in supported:
                return candidate
    return "float32"


def select_device_and_compute_type() -> Tuple[str, str]:
    try:
        import ctranslate2
    except Exception as exc:  # pragma: no cover - should not happen in normal installs
        logger.warning("Failed to import ctranslate2 (%s). Falling back to CPU.", exc)
        return "cpu", "float32"

    device = "cpu"
    if ctranslate2.get_cuda_device_count() > 0:
        device = "cuda"

    if platform.system() == "Darwin":
        try:
            import torch

            if torch.backends.mps.is_available():
                logger.info(
                    "Metal/MPS detected, but faster-whisper uses CTranslate2 which only "
                    "supports CPU/CUDA. Falling back to CPU."
                )
                device = "cpu"
        except Exception:
            pass

    supported = ctranslate2.get_supported_compute_types(device)
    compute_type = _pick_compute_type(device, supported)

    logger.info(
        "Selected device=%s compute_type=%s (supported=%s)",
        device,
        compute_type,
        sorted(supported),
    )
    return device, compute_type
