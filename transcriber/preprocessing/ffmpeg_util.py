import logging
import platform
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from transcriber.utils.constants import FFMPEG_DEFAULT_ARCHIVE_NAME
from transcriber.utils.constants import FFMPEG_PATH
from transcriber.utils.constants import FFMPEG_URL_LINUX_AMD64
from transcriber.utils.constants import FFMPEG_URL_LINUX_ARM64
from transcriber.utils.constants import FFMPEG_URL_MACOS
from transcriber.utils.constants import FFMPEG_URL_WINDOWS

logger = logging.getLogger(__name__)


def _download_file(url: str, dest: Path) -> None:
    logger.info("Downloading ffmpeg from %s", url)
    with urllib.request.urlopen(url) as response, dest.open("wb") as out_file:
        shutil.copyfileobj(response, out_file)


def _extract_ffmpeg(binary_name: str, archive_path: Path, extract_dir: Path) -> Path:
    # Some providers (e.g., Evermeet) return a zip without a .zip suffix.
    # Try zip first, then fall back to tar formats.
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
    except zipfile.BadZipFile:
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(extract_dir)

    for candidate in extract_dir.rglob(binary_name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"ffmpeg binary not found in archive: {archive_path}")


def _detect_download_url(system: str, arch: str) -> tuple[str, str]:
    if system == "Windows":
        if arch not in {"amd64", "x86_64"}:
            raise RuntimeError(f"Unsupported Windows architecture: {arch}")
        return (FFMPEG_URL_WINDOWS, "ffmpeg.exe")

    if system == "Darwin":
        # Evermeet provides macOS binaries via a stable download endpoint.
        return (FFMPEG_URL_MACOS, "ffmpeg")

    if system == "Linux":
        if arch in {"aarch64", "arm64"}:
            return (FFMPEG_URL_LINUX_ARM64, "ffmpeg")
        if arch in {"amd64", "x86_64"}:
            return (FFMPEG_URL_LINUX_AMD64, "ffmpeg")
        raise RuntimeError(f"Unsupported Linux architecture: {arch}")

    raise RuntimeError(f"Unsupported OS: {system}")


def get_local_ffmpeg_path() -> Path:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        resolved = Path(system_ffmpeg).resolve()
        logger.info("Using system ffmpeg at %s", resolved)
        return resolved

    bin_dir = FFMPEG_PATH
    bin_dir.mkdir(parents=True, exist_ok=True)

    system = platform.system()
    arch = platform.machine().lower()
    url, binary_name = _detect_download_url(system, arch)

    target = bin_dir / binary_name
    if target.exists():
        return target

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        archive_name = Path(url).name or FFMPEG_DEFAULT_ARCHIVE_NAME
        archive_path = tmp_dir / archive_name
        _download_file(url, archive_path)
        extracted = _extract_ffmpeg(binary_name, archive_path, tmp_dir)
        shutil.copy2(extracted, target)
        target.chmod(target.stat().st_mode | 0o111)

    logger.info("ffmpeg saved to %s", target)
    return target
