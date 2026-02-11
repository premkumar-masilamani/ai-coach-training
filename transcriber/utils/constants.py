from pathlib import Path

AI_MODEL_PATH = Path("ai_models")
AI_MODEL_FASTER_WHISPER_MEDIUM_REPO = "guillaumekln/faster-whisper-medium"
AI_MODEL_CONFIG = "config.json"
FFMPEG_PATH = Path("tools") / "ffmpeg"
FFMPEG_URL_WINDOWS = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
FFMPEG_URL_MACOS = "https://evermeet.cx/ffmpeg/get/zip"
FFMPEG_URL_LINUX_ARM64 = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz"
FFMPEG_URL_LINUX_AMD64 = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
FFMPEG_DEFAULT_ARCHIVE_NAME = "ffmpeg_download.zip"
DEFAULT_INPUT_DIR = Path("audio_files")
DEFAULT_LANGUAGE = "en"
