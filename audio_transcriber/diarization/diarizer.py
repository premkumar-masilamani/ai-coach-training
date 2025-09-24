import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
from huggingface_hub import snapshot_download
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

from audio_transcriber.utils.constants import AI_MODEL_PATH
from audio_transcriber.utils.time_util import format_timestamp

logger = logging.getLogger(__name__)


class Diarizer:
    model_repo: str = "pyannote/speaker-diarization-3.1"
    model_path: Path = AI_MODEL_PATH / model_repo
    pipeline: Optional[Pipeline] = None

    dependency_model_repos = [
        "pyannote/segmentation-3.0",
        "pyannote/wespeaker-voxceleb-resnet34-LM"
    ]

    def __init__(self, use_auth_token: Optional[str] = None):
        """Initialize the PyannoteDiarizer with local model caching.

        Args:
            use_auth_token (str, optional): Hugging Face token for initial download.
                                          Can also be set via HF_TOKEN environment variable.
        """
        # Get auth token from parameter or environment
        auth_token = use_auth_token or os.getenv('HF_TOKEN')

        if not auth_token:
            logger.warning(
                "No Hugging Face token provided. Set HF_TOKEN environment variable "
                "or pass use_auth_token parameter for initial model download."
            )

        logging.info(f"Using model path: {self.model_path}")

        # Check if model exists locally
        config_file = self.model_path / "config.yaml"
        if not config_file.is_file():
            logger.info("Diarization model not found locally, starting download...")
            self._download_model(auth_token)
        else:
            logger.info("Diarization model found in local cache.")
            # Also check for dependency models
            self._ensure_dependencies(auth_token)

        # Load model from local cache
        self._load_model_from_cache()

    def _download_model(self, auth_token: Optional[str]) -> None:
        """Download the model from Hugging Face Hub to local cache."""
        if not auth_token:
            logger.error(
                "Cannot download model without authentication token. "
                "Please provide HF_TOKEN environment variable or use_auth_token parameter."
            )
            return

        logger.warning(
            "This is a large model and may take a long time to download."
        )

        # Create model directory
        self.model_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading diarization model from Hugging Face Hub: {self.model_repo}")

        try:
            snapshot_download(
                repo_id=self.model_repo,
                local_dir=str(self.model_path),
                local_dir_use_symlinks=False,
                token=auth_token,
            )
            logger.info("Diarization model download complete.")
        except Exception as e:
            logger.error(f"Failed to download diarization model: {e}")
            # Clean up partial download
            if self.model_path.exists():
                import shutil
                shutil.rmtree(self.model_path)
            raise

    def _ensure_dependencies(self, auth_token: Optional[str]) -> None:
        """Ensure all dependency models are downloaded."""
        for dep_model in self.dependency_model_repos:
            dep_path = AI_MODEL_PATH / dep_model
            config_file = dep_path / "config.yaml"

            if not config_file.is_file():
                logger.info(f"Dependency model {dep_model} not found, downloading...")
                self._download_dependency_model(dep_model, auth_token)
            else:
                logger.info(f"Dependency model {dep_model} found in local cache.")

    def _download_dependency_model(self, model_repo: str, auth_token: Optional[str]) -> None:
        """Download a dependency model from Hugging Face Hub to local cache."""
        if not auth_token:
            logger.error(
                f"Cannot download dependency model {model_repo} without authentication token."
            )
            return

        dep_path = Path("models") / model_repo
        dep_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading dependency model: {model_repo}")

        try:
            snapshot_download(
                repo_id=model_repo,
                local_dir=str(dep_path),
                local_dir_use_symlinks=False,
                token=auth_token,
            )
            logger.info(f"Dependency model {model_repo} download complete.")
        except Exception as e:
            logger.error(f"Failed to download dependency model {model_repo}: {e}")
            # Clean up partial download
            if dep_path.exists():
                import shutil
                shutil.rmtree(dep_path)

    def _load_model_from_cache(self) -> None:
        """Load the diarization model from local cache."""
        start_load = time.time()

        try:
            logger.info(f"Loading diarization pipeline from local cache: {self.model_path}")

            # Check if we have a local model
            if self.model_path.exists() and (self.model_path / "config.yaml").exists():
                # Set up proper HuggingFace cache structure
                models_parent = str(self.model_path.parent.parent)  # points to 'models' directory

                # Store original environment variables
                original_vars = {}
                env_vars_to_set = {
                    'HF_HUB_OFFLINE': '1',
                    'TRANSFORMERS_OFFLINE': '1',
                    'HF_HOME': models_parent,
                    'HF_HUB_CACHE': models_parent,
                    'TRANSFORMERS_CACHE': models_parent
                }

                # Save original values and set new ones
                for var_name, var_value in env_vars_to_set.items():
                    original_vars[var_name] = os.environ.get(var_name)
                    os.environ[var_name] = var_value

                try:
                    # Create proper HF cache structure for main model and dependencies
                    hub_cache_dir = Path(models_parent) / "hub"
                    hub_cache_dir.mkdir(exist_ok=True)

                    # Setup cache for main model
                    self._setup_model_cache(hub_cache_dir, self.model_repo, self.model_path)

                    # Setup cache for dependency models
                    for dep_model in self.dependency_model_repos:
                        dep_path = AI_MODEL_PATH / dep_model
                        if dep_path.exists():
                            self._setup_model_cache(hub_cache_dir, dep_model, dep_path)

                    # Load using the repo name with proper offline mode
                    self.pipeline = Pipeline.from_pretrained(
                        self.model_repo,
                        use_auth_token=None,
                        cache_dir=models_parent
                    )

                finally:
                    # Restore original environment variables
                    for var_name, original_value in original_vars.items():
                        if original_value is not None:
                            os.environ[var_name] = original_value
                        elif var_name in os.environ:
                            del os.environ[var_name]
            else:
                # Fallback to online loading if local cache is incomplete
                logger.warning("Local model cache incomplete, falling back to online loading")
                auth_token = os.getenv('HF_TOKEN')
                self.pipeline = Pipeline.from_pretrained(
                    self.model_repo,
                    use_auth_token=auth_token
                )

            # Set device only if pipeline was loaded successfully
            device = torch.device("cpu")  # Default device
            if self.pipeline is not None:
                if torch.backends.mps.is_available():
                    device = torch.device("mps")
                    logger.info("Using Apple Metal Performance Shaders (MPS)")
                elif torch.cuda.is_available():
                    device = torch.device("cuda")
                    logger.info("Using CUDA GPU")
                else:
                    device = torch.device("cpu")
                    logger.info("Using CPU")

                self.pipeline.to(device)

                load_time = time.time() - start_load
                logger.info(f"Diarization pipeline loaded in {load_time:.2f} seconds on {device}")
            else:
                logger.error("Pipeline failed to load")

        except Exception as e:
            logger.error(f"Error loading diarization model from {self.model_path}: {e}")
            self.pipeline = None

    def _setup_model_cache(self, hub_cache_dir: Path, model_repo: str, model_path: Path) -> None:
        """Setup HuggingFace cache structure for a model."""
        repo_cache = hub_cache_dir / f"models--{model_repo.replace('/', '--')}"

        if not repo_cache.exists() and model_path.exists():
            repo_cache.mkdir(parents=True, exist_ok=True)

            # Create snapshots directory structure
            snapshots_dir = repo_cache / "snapshots"
            snapshots_dir.mkdir(exist_ok=True)

            # Create refs directory
            refs_dir = repo_cache / "refs"
            refs_dir.mkdir(exist_ok=True)

            # Create a fake revision directory
            fake_revision = "main"
            revision_dir = snapshots_dir / fake_revision

            if not revision_dir.exists():
                # Create symlinks from cache to actual model files
                revision_dir.mkdir(exist_ok=True)
                for item in model_path.iterdir():
                    if item.is_file():
                        link_target = revision_dir / item.name
                        if not link_target.exists():
                            try:
                                if hasattr(os, 'symlink'):
                                    os.symlink(str(item), str(link_target))
                                else:
                                    import shutil
                                    shutil.copy2(str(item), str(link_target))
                            except (OSError, ImportError):
                                # Fallback to copying if symlink fails
                                import shutil
                                shutil.copy2(str(item), str(link_target))

                # Create refs/main file pointing to our fake revision
                with open(refs_dir / "main", "w") as f:
                    f.write(fake_revision)

    def _convert_audio_format(self, audio_path: Path) -> Tuple[Path, bool]:
        """Convert audio file to WAV format if needed.

        Args:
            audio_path (str): Path to the input audio file.

        Returns:
            Tuple[str, bool]: (converted_file_path, is_temporary)
                - converted_file_path: Path to the converted file (or original if no conversion needed)
                - is_temporary: True if the converted file is temporary and should be cleaned up
        """
        file_extension = audio_path.suffix.lower()

        # If already a supported format, return as-is
        if file_extension in ['.wav', '.flac', '.mp3']:
            return audio_path, False

        # For unsupported formats, try to convert using ffmpeg first, then pydub as fallback
        temp_wav_path = None

        # Try ffmpeg first
        try:
            import subprocess

            # Create temporary WAV file
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_wav_path = temp_wav.name
            temp_wav.close()

            logger.info(f"Converting {file_extension} to WAV format using ffmpeg...")

            # Use ffmpeg to convert to WAV
            cmd = [
                'ffmpeg', '-i', audio_path,
                '-acodec', 'pcm_s16le',  # 16-bit PCM
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite output file
                temp_wav_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"Successfully converted {file_extension} to WAV format using ffmpeg")
                return temp_wav_path, True
            else:
                logger.warning(f"FFmpeg conversion failed: {result.stderr}")
                # Clean up temp file and try pydub fallback
                try:
                    os.unlink(temp_wav_path)
                except:
                    pass
                temp_wav_path = None

        except FileNotFoundError:
            logger.info("FFmpeg not found, trying pydub as fallback...")
        except Exception as e:
            logger.warning(f"FFmpeg conversion failed: {e}, trying pydub as fallback...")
            if temp_wav_path:
                try:
                    os.unlink(temp_wav_path)
                except:
                    pass
                temp_wav_path = None

        # Try pydub as fallback if available
        try:
            from pydub import AudioSegment

            # Create temporary WAV file
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_wav_path = temp_wav.name
            temp_wav.close()

            logger.info(f"Converting {file_extension} to WAV format using pydub...")

            # Load audio file with pydub
            audio = AudioSegment.from_file(audio_path)

            # Convert to mono, 16kHz, 16-bit
            audio = audio.set_channels(1)  # Mono
            audio = audio.set_frame_rate(16000)  # 16kHz
            audio = audio.set_sample_width(2)  # 16-bit

            # Export as WAV
            audio.export(temp_wav_path, format="wav")

            logger.info(f"Successfully converted {file_extension} to WAV format using pydub")
            return temp_wav_path, True

        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"Pydub not available for audio conversion: {e}")
            logger.warning(
                "Audio conversion failed. Please ensure ffmpeg is properly installed "
                "or try converting the file to WAV format manually before processing."
            )
            return audio_path, False
        except Exception as e:
            logger.error(f"Pydub conversion also failed: {e}")
            if temp_wav_path:
                try:
                    os.unlink(temp_wav_path)
                except:
                    pass
            logger.warning(
                "All audio conversion methods failed. You may need to convert the M4A file "
                "to WAV format manually using: ffmpeg -i input.m4a -acodec pcm_s16le "
                "-ar 16000 -ac 1 output.wav"
            )
            return audio_path, False

    def _format_diarization_output(self, diarization) -> str:
        """Format diarization results into a readable string.

        Args:
            diarization: The diarization result from the pipeline.

        Returns:
            str: Formatted diarization output with timestamps and speaker labels.
        """
        logger.info("Formatting diarization output.")
        if diarization is None:
            logger.warning("Cannot format output: diarization result is None.")
            return "No diarization results available."

        try:
            lines = []
            logger.info("Iterating through diarization tracks...")
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                line = {
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                }
                lines.append(line)
                logger.debug(f"Formatted segment: {line}")
            diarized_json = {"diarization": lines}

            logger.info(f"Diarization output formatted with {len(lines)} segments.")
            return json.dumps(diarized_json)

        except Exception as e:
            logger.error(f"Error formatting diarization output: {e}", exc_info=True)
            return "Error formatting diarization results."

    def diarize(self, audio_path: Path):
        """Perform speaker diarization on the given audio file.

        Args:
            audio_path (str): Path to the audio file to diarize.

        Returns:
            Diarization result with speaker segments or None if diarization fails.
        """
        logger.info(f"Diarization process started for: {audio_path}")
        start_time = time.time()

        if self.pipeline is None:
            logger.error("Diarization pipeline is not available. Model may not be loaded correctly.")
            return None

        logger.info("Diarization pipeline is available.")

        # Convert audio format if needed
        # TODO: Move this to pre-processing
        converted_audio_path, is_temporary = self._convert_audio_format(audio_path)
        logger.info(f"Starting diarization for: {audio_path}")
        if converted_audio_path != audio_path:
            logger.info(f"Using converted audio file: {converted_audio_path}")

        try:
            logger.info("Applying diarization pipeline...")
            # Perform diarization with progress tracking
            with ProgressHook() as hook:
                diarization = self.pipeline(converted_audio_path, hook=hook)
                logger.debug(f"Diarization result: {diarization}")

            logger.info("Diarization pipeline applied successfully.")
            formatted_output = self._format_diarization_output(diarization)
            logger.info(f"Diarization completed in {time.time() - start_time:.2f} seconds for: {audio_path}")
            return formatted_output
        except Exception as e:
            logger.error(f"Error during diarization of {audio_path}: {e}", exc_info=True)
            return None
        finally:
            if is_temporary and converted_audio_path and os.path.exists(converted_audio_path):
                try:
                    os.remove(converted_audio_path)
                    logger.info(f"Removed temporary audio file: {converted_audio_path}")
                except OSError as e:
                    logger.error(f"Error removing temporary file {converted_audio_path}: {e}")

    def get_speaker_count(self, diarization) -> int:
        """Get the number of unique speakers detected.

        Args:
            diarization: The diarization result from the pipeline.

        Returns:
            int: Number of unique speakers detected.
        """
        if diarization is None:
            return 0

        try:
            speakers = set()
            for _, _, speaker in diarization.itertracks(yield_label=True):
                speakers.add(speaker)
            return len(speakers)
        except Exception as e:
            logger.error(f"Error counting speakers: {e}")
            return 0
