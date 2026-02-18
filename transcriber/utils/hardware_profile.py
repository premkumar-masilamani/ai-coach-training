import ctypes
import logging
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HardwareProfile:
    system: str
    architecture: str
    ram_gb: int
    cpu_cores: int
    accelerator: str
    has_gpu: bool
    ram_bucket: str
    processing_score: int


@dataclass(frozen=True)
class WhisperBackend:
    name: str
    cmake_flags: tuple[str, ...]


def _run_probe(command: list[str], timeout_seconds: int = 3) -> tuple[int, str]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        return result.returncode, (result.stdout or "").strip()
    except Exception:
        return 1, ""


def _detect_windows_total_ram_bytes() -> int:
    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return int(status.ullTotalPhys)
    return 0


def _detect_total_ram_bytes(system: str) -> int:
    if system == "Windows":
        return _detect_windows_total_ram_bytes()

    if hasattr(os, "sysconf"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            page_count = int(os.sysconf("SC_PHYS_PAGES"))
            if page_size > 0 and page_count > 0:
                return page_size * page_count
        except (KeyError, ValueError, OSError):
            pass

    if system == "Darwin":
        code, output = _run_probe(["sysctl", "-n", "hw.memsize"])
        if code == 0 and output.isdigit():
            return int(output)

    return 0


def _ram_bucket(ram_gb: int) -> str:
    if ram_gb >= 32:
        return "32GB+"
    if ram_gb >= 16:
        return "16GB"
    if ram_gb >= 8:
        return "8GB"
    return "<8GB"


def _has_nvidia_gpu() -> bool:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False
    code, output = _run_probe([nvidia_smi, "-L"], timeout_seconds=5)
    return code == 0 and "GPU" in output


def _has_vulkan_runtime() -> bool:
    vulkaninfo = shutil.which("vulkaninfo")
    if not vulkaninfo:
        return False
    code, output = _run_probe([vulkaninfo, "--summary"], timeout_seconds=5)
    if code == 0 and output:
        return True
    code, _ = _run_probe([vulkaninfo], timeout_seconds=5)
    return code == 0


def _detect_accelerator(system: str) -> str:
    if _has_nvidia_gpu():
        return "cuda"
    if system == "Darwin":
        return "metal"
    if system in {"Linux", "Windows"} and _has_vulkan_runtime():
        return "vulkan"
    return "cpu"


def _processing_score(cpu_cores: int, ram_gb: int, accelerator: str) -> int:
    score = min(max(cpu_cores, 1), 24)
    if accelerator == "cuda":
        score += 20
    elif accelerator == "metal":
        score += 14
    elif accelerator == "vulkan":
        score += 10

    if ram_gb >= 32:
        score += 8
    elif ram_gb >= 16:
        score += 4
    elif ram_gb >= 8:
        score += 2

    return score


@lru_cache(maxsize=1)
def detect_hardware_profile() -> HardwareProfile:
    system = platform.system()
    architecture = platform.machine().lower()
    total_ram_bytes = _detect_total_ram_bytes(system)
    ram_gb = max(1, int(total_ram_bytes / (1024**3)))
    cpu_cores = os.cpu_count() or 1
    accelerator = _detect_accelerator(system)
    ram_bucket = _ram_bucket(ram_gb)
    score = _processing_score(cpu_cores, ram_gb, accelerator)

    profile = HardwareProfile(
        system=system,
        architecture=architecture,
        ram_gb=ram_gb,
        cpu_cores=cpu_cores,
        accelerator=accelerator,
        has_gpu=accelerator in {"cuda", "metal", "vulkan"},
        ram_bucket=ram_bucket,
        processing_score=score,
    )

    logger.info(
        "Hardware profile detected: os=%s arch=%s ram=%sGB bucket=%s cpu_cores=%s accelerator=%s score=%s",
        profile.system,
        profile.architecture,
        profile.ram_gb,
        profile.ram_bucket,
        profile.cpu_cores,
        profile.accelerator,
        profile.processing_score,
    )
    return profile


def select_whisper_backend(profile: HardwareProfile) -> WhisperBackend:
    if profile.accelerator == "cuda":
        return WhisperBackend(name="cuda", cmake_flags=("-DGGML_CUDA=ON",))
    if profile.accelerator == "metal":
        return WhisperBackend(name="metal", cmake_flags=("-DGGML_METAL=ON",))
    if profile.accelerator == "vulkan":
        return WhisperBackend(name="vulkan", cmake_flags=("-DGGML_VULKAN=ON",))
    return WhisperBackend(name="cpu", cmake_flags=())


def cpu_backend_reset_flags() -> tuple[str, ...]:
    return ("-DGGML_CUDA=OFF", "-DGGML_METAL=OFF", "-DGGML_VULKAN=OFF")
