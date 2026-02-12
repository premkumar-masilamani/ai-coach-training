from pathlib import Path

AI_MODEL_PATH = Path("ai_models")
AI_MODEL_WHISPER_CPP_PATH = AI_MODEL_PATH / "whisper.cpp"
AI_MODEL_WHISPER_CPP_DEFAULT_MODEL = AI_MODEL_WHISPER_CPP_PATH / "ggml-base.en.bin"
AI_MODEL_WHISPER_CPP_DEFAULT_MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
FFMPEG_PATH = Path("tools") / "ffmpeg"
WHISPER_CPP_PATH = Path("tools") / "whisper.cpp"
WHISPER_CPP_LOCAL_BIN = WHISPER_CPP_PATH / "build" / "bin" / "whisper-cli"
WHISPER_CPP_LOCAL_LEGACY_BIN = WHISPER_CPP_PATH / "main"
WHISPER_CPP_REPO_MODEL = WHISPER_CPP_PATH / "models" / "ggml-base.en.bin"
FFMPEG_URL_WINDOWS = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
FFMPEG_URL_MACOS = "https://evermeet.cx/ffmpeg/get/zip"
FFMPEG_URL_LINUX_ARM64 = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz"
FFMPEG_URL_LINUX_AMD64 = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
FFMPEG_DEFAULT_ARCHIVE_NAME = "ffmpeg_download.zip"
DEFAULT_INPUT_DIR = Path("audio_files")
DEFAULT_LANGUAGE = "en"
