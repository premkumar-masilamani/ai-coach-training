import os
import platform
from pathlib import Path

def _resolve_coachlens_home() -> Path:
    system = platform.system()
    home = Path.home()

    if system == "Windows":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "CoachLens"
        return home / "AppData" / "Local" / "CoachLens"

    if system == "Linux":
        xdg_data_home = os.environ.get("XDG_DATA_HOME")
        if xdg_data_home:
            return Path(xdg_data_home) / "coachlens"
        return home / ".local" / "share" / "coachlens"

    # macOS default
    return home / ".coachlens"


COACHLENS_HOME = _resolve_coachlens_home()
AI_MODEL_PATH = COACHLENS_HOME / "models"
TOOLS_PATH = COACHLENS_HOME / "tools"
REPOS_PATH = COACHLENS_HOME / "repos"
AI_MODEL_WHISPER_CPP_PATH = AI_MODEL_PATH / "whisper.cpp"
WHISPER_CPP_MODEL_URL_PREFIX = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/"
AI_MODEL_WHISPER_CPP_DEFAULT_MODEL = AI_MODEL_WHISPER_CPP_PATH / "ggml-base.en.bin"
AI_MODEL_WHISPER_CPP_DEFAULT_MODEL_URL = f"{WHISPER_CPP_MODEL_URL_PREFIX}ggml-base.en.bin"
FFMPEG_PATH = TOOLS_PATH / "ffmpeg"
WHISPER_CPP_PATH = REPOS_PATH / "whisper.cpp"
WHISPER_CPP_LOCAL_BIN = WHISPER_CPP_PATH / "build" / "bin" / (
    "whisper-cli.exe" if platform.system() == "Windows" else "whisper-cli"
)
WHISPER_CPP_LOCAL_LEGACY_BIN = WHISPER_CPP_PATH / (
    "main.exe" if platform.system() == "Windows" else "main"
)
WHISPER_CPP_REPO_MODEL_DIR = WHISPER_CPP_PATH / "models"
FFMPEG_URL_WINDOWS = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
FFMPEG_URL_MACOS = "https://evermeet.cx/ffmpeg/get/zip"
FFMPEG_URL_LINUX_ARM64 = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz"
FFMPEG_URL_LINUX_AMD64 = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
FFMPEG_DEFAULT_ARCHIVE_NAME = "ffmpeg_download.zip"
DEFAULT_INPUT_DIR = Path.home() / "Downloads"
DEFAULT_LANGUAGE = "en"
