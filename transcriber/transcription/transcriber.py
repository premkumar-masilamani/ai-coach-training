import json
import logging
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from threading import Event
from threading import Thread
from typing import Callable, Optional

from transcriber.utils.constants import DEFAULT_LANGUAGE
from transcriber.utils.constants import WHISPER_CPP_LOCAL_BIN
from transcriber.utils.constants import WHISPER_CPP_LOCAL_LEGACY_BIN
from transcriber.utils.constants import WHISPER_CPP_PATH
from transcriber.utils.hardware_profile import cpu_backend_reset_flags
from transcriber.utils.hardware_profile import detect_hardware_profile
from transcriber.utils.hardware_profile import select_whisper_backend
from transcriber.utils.model_selection import local_model_candidates
from transcriber.utils.model_selection import min_ram_for_model
from transcriber.utils.model_selection import repo_model_candidates
from transcriber.utils.model_selection import select_model_for_hardware
from transcriber.utils.time_util import format_timestamp

logger = logging.getLogger(__name__)


def _parse_timestamp_to_seconds(raw: str) -> float:
    text = raw.strip()
    if not text:
        return 0.0
    text = text.replace(",", ".")
    parts = text.split(":")
    if len(parts) != 3:
        return 0.0
    hours = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    return (hours * 3600.0) + (minutes * 60.0) + seconds


def _extract_segments(payload: dict) -> list[dict]:
    segments = payload.get("transcription")
    if not isinstance(segments, list):
        segments = payload.get("segments")
    if not isinstance(segments, list):
        result = payload.get("result")
        if isinstance(result, dict):
            segments = result.get("segments")
    if not isinstance(segments, list):
        return []

    lines = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text", "")).strip()
        if not text:
            continue

        start_raw = segment.get("start")
        end_raw = segment.get("end")

        if isinstance(start_raw, (int, float)) and isinstance(end_raw, (int, float)):
            start_sec = float(start_raw)
            end_sec = float(end_raw)
        else:
            offsets = segment.get("offsets", {})
            if isinstance(offsets, dict):
                from_ms = offsets.get("from")
                to_ms = offsets.get("to")
                if isinstance(from_ms, (int, float)) and isinstance(to_ms, (int, float)):
                    start_sec = float(from_ms) / 1000.0
                    end_sec = float(to_ms) / 1000.0
                else:
                    timestamps = segment.get("timestamps", {})
                    start_sec = _parse_timestamp_to_seconds(str(timestamps.get("from", "")))
                    end_sec = _parse_timestamp_to_seconds(str(timestamps.get("to", "")))
            else:
                start_sec = 0.0
                end_sec = 0.0

        lines.append(
            {
                "start": format_timestamp(start_sec),
                "end": format_timestamp(end_sec),
                "text": text,
            }
        )

    return lines


class Transcriber:
    def __init__(self, progress_cb: Optional[Callable[[str, str, str], None]] = None):
        self.repo_path: Path = WHISPER_CPP_PATH
        self.progress_cb = progress_cb
        self.hardware_profile = detect_hardware_profile()
        self.backend = select_whisper_backend(self.hardware_profile)
        self.model_spec = select_model_for_hardware(self.hardware_profile)
        self.model_min_ram_gb = min_ram_for_model(self.model_spec.model_id)
        self.model_path: Path = self.model_spec.local_path

        logger.info(
            "Selected whisper model=%s format=%s (min_ram=%sGB) for os=%s arch=%s ram_bucket=%s score=%s",
            self.model_spec.model_id,
            self.model_spec.format,
            self.model_min_ram_gb,
            self.hardware_profile.system,
            self.hardware_profile.architecture,
            self.hardware_profile.ram_bucket,
            self.hardware_profile.processing_score,
        )

        if self.hardware_profile.ram_gb < self.model_min_ram_gb:
            logger.warning(
                "System RAM (%sGB) is below recommended minimum (%sGB) for model %s.",
                self.hardware_profile.ram_gb,
                self.model_min_ram_gb,
                self.model_spec.model_id,
            )

        if self.repo_path.exists():
            self._report("tool.repo", "Ready", str(self.repo_path))
        else:
            self._report("tool.repo", "Missing", str(self.repo_path))
        self.binary_path: Optional[str] = self._resolve_whisper_cpp_binary()

        if not self.binary_path:
            self._report("tool.repo", "Checking", str(self.repo_path))
            self._report("tool.binary", "Missing", str(WHISPER_CPP_LOCAL_BIN))
            logger.info("whisper.cpp binary not found. Bootstrapping whisper.cpp on first run.")
            self._bootstrap_whisper_cpp()
            self.binary_path = self._resolve_whisper_cpp_binary()

        if not self.binary_path:
            logger.error(
                "whisper.cpp bootstrap failed. Expected binary at %s or in PATH.",
                WHISPER_CPP_LOCAL_BIN,
            )
            self._report("tool.binary", "Failed", str(WHISPER_CPP_LOCAL_BIN))
        else:
            logger.info("Using whisper.cpp binary: %s", self.binary_path)
            self._report("tool.binary", "Ready", self.binary_path)

        if not self.model_path.is_file():
            for repo_candidate in repo_model_candidates(self.model_spec):
                if repo_candidate.is_file():
                    self.model_path = repo_candidate
                    break

        if not self.model_path.is_file():
            self._report("model.default", "Missing", str(self.model_path))
            logger.info(
                "whisper.cpp model %s not found. Downloading on first run (preferred format=%s).",
                self.model_spec.model_id,
                self.model_spec.format,
            )
            self._download_model(
                destination=self.model_path,
                source_url=self.model_spec.download_url,
                model_id=self.model_spec.model_id,
            )

        if not self.model_path.is_file():
            for local_candidate in local_model_candidates(self.model_spec):
                if local_candidate.is_file():
                    self.model_path = local_candidate
                    break
        if not self.model_path.is_file():
            for repo_candidate in repo_model_candidates(self.model_spec):
                if repo_candidate.is_file():
                    self.model_path = repo_candidate
                    break

        if self.model_path.is_file():
            logger.info("Using whisper.cpp model: %s", self.model_path)
            self._report("model.default", "Ready", str(self.model_path))
        else:
            logger.error("Failed to obtain whisper.cpp model file.")
            self._report("model.default", "Failed", str(self.model_path))

    def _report(self, item_id: str, status: str, path_text: str):
        if self.progress_cb:
            try:
                self.progress_cb(item_id, status, path_text)
            except Exception:
                logger.debug("Progress callback failed for %s", item_id)

    def _resolve_whisper_cpp_binary(self) -> Optional[str]:
        if WHISPER_CPP_LOCAL_BIN.is_file():
            return str(WHISPER_CPP_LOCAL_BIN)
        if WHISPER_CPP_LOCAL_LEGACY_BIN.is_file():
            return str(WHISPER_CPP_LOCAL_LEGACY_BIN)

        for binary_name in ("whisper-cli", "whisper-cli.exe", "main", "main.exe"):
            from_path = shutil.which(binary_name)
            if from_path:
                return from_path

        return None

    def _run_command(self, command: list[str], cwd: Optional[Path] = None) -> bool:
        try:
            subprocess.run(
                command,
                cwd=str(cwd) if cwd else None,
                check=True,
                capture_output=True,
                text=True,
            )
            return True
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            details = stderr or stdout or str(exc)
            logger.error("Command failed: %s\n%s", " ".join(command), details)
            return False
        except Exception as exc:
            logger.error("Command error: %s (%s)", " ".join(command), exc)
            return False

    def _configure_whisper_cpp_build(self, cmake_bin: str) -> bool:
        backend_flags = [*cpu_backend_reset_flags(), *self.backend.cmake_flags]
        logger.info(
            "Configuring whisper.cpp backend=%s with flags: %s",
            self.backend.name,
            " ".join(backend_flags) if backend_flags else "<none>",
        )
        configured = self._run_command(
            [cmake_bin, "-S", ".", "-B", "build", *backend_flags],
            cwd=self.repo_path,
        )
        if configured:
            return True

        if self.backend.name == "cpu":
            return False

        logger.warning(
            "Failed to configure whisper.cpp with backend=%s. Retrying with CPU-only build.",
            self.backend.name,
        )
        return self._run_command(
            [cmake_bin, "-S", ".", "-B", "build", *cpu_backend_reset_flags()],
            cwd=self.repo_path,
        )

    def _bootstrap_whisper_cpp(self):
        git_bin = shutil.which("git")
        cmake_bin = shutil.which("cmake")
        if not git_bin:
            logger.error("Cannot bootstrap whisper.cpp: `git` is not installed.")
            self._report("tool.repo", "Failed", str(self.repo_path))
            return
        if not cmake_bin:
            logger.error("Cannot bootstrap whisper.cpp: `cmake` is not installed.")
            self._report("tool.binary", "Failed", str(WHISPER_CPP_LOCAL_BIN))
            return

        if not self.repo_path.exists():
            self.repo_path.parent.mkdir(parents=True, exist_ok=True)
            self._report("tool.repo", "Downloading", str(self.repo_path))
            logger.info("Cloning whisper.cpp into %s", self.repo_path)
            ok = self._run_command(
                [git_bin, "clone", "--depth", "1", "https://github.com/ggml-org/whisper.cpp.git", str(self.repo_path)]
            )
            if not ok:
                self._report("tool.repo", "Failed", str(self.repo_path))
                return
            self._report("tool.repo", "Ready", str(self.repo_path))
        else:
            self._report("tool.repo", "Ready", str(self.repo_path))

        self._report("tool.binary", "Building", str(WHISPER_CPP_LOCAL_BIN))
        logger.info("Building whisper.cpp (preferred backend: %s).", self.backend.name)
        configured = self._configure_whisper_cpp_build(cmake_bin)
        if not configured:
            self._report("tool.binary", "Failed", str(WHISPER_CPP_LOCAL_BIN))
            return

        built = self._run_command(
            [cmake_bin, "--build", "build", "-j"],
            cwd=self.repo_path,
        )
        if not built:
            self._report("tool.binary", "Failed", str(WHISPER_CPP_LOCAL_BIN))
            return

        self._report("tool.binary", "Ready", str(WHISPER_CPP_LOCAL_BIN))

    def _download_model(
        self,
        destination: Path,
        source_url: str,
        model_id: str,
    ) -> bool:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_path = destination.with_suffix(destination.suffix + ".tmp")
        self._report("model.default", "Downloading", str(destination))
        logger.info("Downloading whisper.cpp model %s to %s", model_id, destination)
        try:
            with urllib.request.urlopen(source_url, timeout=120) as response:
                with open(temp_path, "wb") as out:
                    shutil.copyfileobj(response, out)
            temp_path.replace(destination)
            self._report("model.default", "Ready", str(destination))
            return True
        except Exception as exc:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            logger.error("Failed to download whisper.cpp model %s from %s: %s", model_id, source_url, exc)
            return False

    def transcribe(self, audio_file: Path, stop_event: Optional[Event] = None) -> Optional[str]:
        if stop_event and stop_event.is_set():
            raise InterruptedError(f"Transcription canceled for {audio_file}")

        if not self.binary_path:
            logger.error("Cannot transcribe: whisper.cpp binary is not available.")
            return None
        if not self.model_path.is_file():
            logger.error("Cannot transcribe: whisper.cpp model file is missing: %s", self.model_path)
            return None

        logger.info("Transcribing with whisper.cpp: %s", audio_file)
        start_transcribe = time.time()

        with tempfile.TemporaryDirectory(prefix="whispercpp_") as temp_dir:
            output_base = Path(temp_dir) / "transcript"
            command = [
                self.binary_path,
                "-m",
                str(self.model_path),
                "-f",
                str(audio_file),
                "-l",
                DEFAULT_LANGUAGE,
                "-oj",
                "-of",
                str(output_base),
            ]
            logger.debug("whisper.cpp command: %s", " ".join(command))

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            stdout_lines: list[str] = []
            stderr_lines: list[str] = []

            def _drain_stream(stream, sink: list[str], stream_name: str):
                for line in iter(stream.readline, ""):
                    text = line.rstrip()
                    sink.append(text)
                    if text:
                        logger.info("whisper.cpp %s: %s", stream_name, text)
                stream.close()

            stdout_thread = Thread(
                target=_drain_stream, args=(process.stdout, stdout_lines, "stdout"), daemon=True
            )
            stderr_thread = Thread(
                target=_drain_stream, args=(process.stderr, stderr_lines, "stderr"), daemon=True
            )
            stdout_thread.start()
            stderr_thread.start()

            while process.poll() is None:
                if stop_event and stop_event.is_set():
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise InterruptedError(f"Transcription canceled for {audio_file}")
                time.sleep(0.1)

            stdout_thread.join(timeout=2)
            stderr_thread.join(timeout=2)
            stdout = "\n".join(stdout_lines).strip()
            stderr = "\n".join(stderr_lines).strip()

            if process.returncode != 0:
                details = (stderr or stdout or "Unknown error").strip()
                logger.error("whisper.cpp failed for %s: %s", audio_file, details)
                return None

            json_output = output_base.with_suffix(".json")
            if not json_output.is_file():
                logger.error("whisper.cpp did not produce JSON output for %s", audio_file)
                return None

            try:
                payload = json.loads(json_output.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.error("Failed to parse whisper.cpp JSON output for %s: %s", audio_file, exc)
                return None

            segments = _extract_segments(payload)
            if not segments:
                logger.warning("No segments found in whisper.cpp output for %s", audio_file)
                return None

        logger.info(
            "Transcription completed in %.2f seconds.",
            time.time() - start_transcribe,
        )
        return json.dumps({"transcription": segments})
