import logging
import os
import time
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from huggingface_hub import snapshot_download

from audio_transcriber.utils.constants import AI_MODEL_PATH, DEFAULT_DEVICE_CPU
from audio_transcriber.utils.time_util import format_timestamp

logger = logging.getLogger(__name__)

class Diarizer:
    # Model repo name
    model_repo: str = "pyannote/speaker-diarization-3.1"
    # Model instance
    pipeline: Optional[Pipeline] = None

    # Dependency models required for diarization
    dependency_models = [
        "pyannote/segmentation-3.0",
        "pyannote/wespeaker-voxceleb-resnet34-LM"
    ]

    def __init__(self, use_auth_token: Optional[str] = None, offline: bool = False):
        """Initialize the Diarizer with local model caching.

        Args:
            use_auth_token (str, optional): Hugging Face token for initial download.
                                          Can also be set via HF_TOKEN environment variable.
            offline (bool): If True, strictly use local models and do not attempt to download.
        """
        self.offline = offline
        self.model_path = AI_MODEL_PATH / self.model_repo

        if not self.offline:
            # Get auth token from parameter or environment
            auth_token = use_auth_token or os.getenv('HF_TOKEN')

            # Check if main model exists locally
            config_file = self.model_path / "config.yaml"
            if not config_file.is_file():
                if not auth_token:
                    logger.warning(
                        "Diarization model not found locally and no auth token provided. "
                        "Offline mode is disabled, but download will fail."
                    )
                logger.info("Diarization model not found locally, starting download...")
                self.download_models(auth_token)
            else:
                # Also check for dependency models
                dependencies_complete = True
                for dep_model in self.dependency_models:
                    dep_path = AI_MODEL_PATH / dep_model
                    if not (dep_path / "config.yaml").is_file():
                        dependencies_complete = False
                        break

                if not dependencies_complete:
                    logger.info("Dependency models incomplete, starting download...")
                    self.download_models(auth_token)
        else:
            logger.info("Running in offline mode. Strictly using local models.")

        # Load model from local cache
        self._load_model_from_cache()

    def download_models(self, auth_token: Optional[str] = None) -> None:
        """Download the main model and all dependencies from Hugging Face Hub.

        Args:
            auth_token (str, optional): Hugging Face token.
        """
        token = auth_token or os.getenv('HF_TOKEN')
        if not token:
            logger.error(
                "Cannot download models without authentication token. "
                "Please provide HF_TOKEN environment variable or use_auth_token parameter."
            )
            return

        logger.warning("Downloading diarization models. This may take a long time.")

        # Download main model
        self._download_single_model(self.model_repo, self.model_path, token)

        # Download dependency models
        for dep_model in self.dependency_models:
            dep_path = AI_MODEL_PATH / dep_model
            self._download_single_model(dep_model, dep_path, token)

    def _download_single_model(self, repo_id: str, local_dir: Path, token: str) -> None:
        """Download a single model from Hugging Face Hub."""
        logger.info(f"Downloading model: {repo_id}")
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                token=token,
            )
            logger.info(f"Download complete: {repo_id}")
        except Exception as e:
            logger.error(f"Failed to download {repo_id}: {e}")
            if local_dir.exists():
                shutil.rmtree(local_dir)
            raise

    def _load_model_from_cache(self) -> None:
        """Load the diarization model from local cache using a fake HF hub structure."""
        start_load = time.time()

        try:
            logger.info(f"Loading diarization pipeline from: {self.model_path}")

            # Check if we have a local model
            if self.model_path.exists() and (self.model_path / "config.yaml").exists():
                models_parent = str(AI_MODEL_PATH)
                hub_cache_dir = AI_MODEL_PATH / "hub"
                hub_cache_dir.mkdir(exist_ok=True, parents=True)

                # Store original environment variables to restore them later
                original_vars = {
                    var: os.environ.get(var)
                    for var in ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HOME', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE']
                }

                # Set environment variables for offline loading
                os.environ['HF_HUB_OFFLINE'] = '1'
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                os.environ['HF_HOME'] = models_parent
                os.environ['HF_HUB_CACHE'] = models_parent
                os.environ['TRANSFORMERS_CACHE'] = models_parent

                try:
                    # Setup cache for main model
                    self._setup_model_cache(hub_cache_dir, self.model_repo, self.model_path)

                    # Setup cache for dependency models
                    for dep_model in self.dependency_models:
                        dep_path = AI_MODEL_PATH / dep_model
                        if dep_path.exists():
                            self._setup_model_cache(hub_cache_dir, dep_model, dep_path)

                    # Load using the repo name. With the env vars and fake cache,
                    # it will load from AI_MODEL_PATH/hub
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
                if self.offline:
                    logger.error("Local model not found and offline mode is enabled. Cannot load pipeline.")
                    return

                # Fallback to online loading if local cache is incomplete and not in offline mode
                logger.warning("Local model cache incomplete, falling back to online loading")
                auth_token = os.getenv('HF_TOKEN')
                self.pipeline = Pipeline.from_pretrained(
                    self.model_repo,
                    use_auth_token=auth_token
                )

            # Set device
            if self.pipeline is not None:
                device = torch.device(DEFAULT_DEVICE_CPU)
                if torch.backends.mps.is_available():
                    device = torch.device("mps")
                    logger.info("Using Apple Metal Performance Shaders (MPS)")
                elif torch.cuda.is_available():
                    device = torch.device("cuda")
                    logger.info("Using CUDA GPU")
                else:
                    logger.info("Using CPU")

                self.pipeline.to(device)
                load_time = time.time() - start_load
                logger.info(f"Diarization pipeline loaded in {load_time:.2f} seconds on {device}")
            else:
                logger.error("Pipeline failed to load")

        except Exception as e:
            logger.error(f"Error loading diarization model: {e}")
            self.pipeline = None

    def _setup_model_cache(self, hub_cache_dir: Path, model_repo: str, model_path: Path) -> None:
        """Setup HuggingFace cache structure for a model to trick from_pretrained into offline mode."""
        repo_cache = hub_cache_dir / f"models--{model_repo.replace('/', '--')}"

        if not repo_cache.exists() and model_path.exists():
            repo_cache.mkdir(parents=True, exist_ok=True)

            snapshots_dir = repo_cache / "snapshots"
            snapshots_dir.mkdir(exist_ok=True)

            refs_dir = repo_cache / "refs"
            refs_dir.mkdir(exist_ok=True)

            # Use a fixed revision name
            fake_revision = "main"
            revision_dir = snapshots_dir / fake_revision

            if not revision_dir.exists():
                revision_dir.mkdir(exist_ok=True)
                for item in model_path.iterdir():
                    if item.is_file():
                        link_target = revision_dir / item.name
                        if not link_target.exists():
                            try:
                                if hasattr(os, 'symlink'):
                                    os.symlink(str(item), str(link_target))
                                else:
                                    shutil.copy2(str(item), str(link_target))
                            except (OSError, ImportError):
                                shutil.copy2(str(item), str(link_target))

                with open(refs_dir / "main", "w") as f:
                    f.write(fake_revision)

    def _convert_audio_format(self, audio_path: str) -> Tuple[str, bool]:
        """Convert audio file to WAV format if needed using ffmpeg or pydub."""
        audio_path_obj = Path(audio_path)
        file_extension = audio_path_obj.suffix.lower()

        if file_extension in ['.wav', '.flac', '.mp3']:
            return audio_path, False

        temp_wav_path = None
        try:
            import subprocess
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_wav_path = temp_wav.name
            temp_wav.close()

            logger.info(f"Converting {file_extension} to WAV using ffmpeg...")
            cmd = ['ffmpeg', '-i', audio_path, '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', temp_wav_path]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                return temp_wav_path, True
            else:
                logger.warning(f"FFmpeg failed: {result.stderr}")
                os.unlink(temp_wav_path)
                temp_wav_path = None
        except Exception:
            if temp_wav_path:
                os.unlink(temp_wav_path)
                temp_wav_path = None

        try:
            from pydub import AudioSegment
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_wav_path = temp_wav.name
            temp_wav.close()

            logger.info(f"Converting {file_extension} to WAV using pydub...")
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
            audio.export(temp_wav_path, format="wav")
            return temp_wav_path, True
        except Exception as e:
            if temp_wav_path:
                os.unlink(temp_wav_path)
            logger.error(f"Conversion failed: {e}")
            return audio_path, False

    def diarize(self, audio_path: str, min_speakers: Optional[int] = None, max_speakers: Optional[int] = None):
        """Perform speaker diarization on the given audio file."""
        if self.pipeline is None:
            logger.error("Diarization pipeline is not available.")
            return None

        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return None

        converted_audio_path, is_temporary = self._convert_audio_format(audio_path)

        try:
            logger.info(f"Starting diarization for: {audio_path}")
            start_time = time.time()

            params = {}
            if min_speakers is not None:
                params['min_speakers'] = min_speakers
            if max_speakers is not None:
                params['max_speakers'] = max_speakers

            with ProgressHook() as hook:
                diarization = self.pipeline(converted_audio_path, hook=hook, **params)

            logger.info(f"Diarization completed in {time.time() - start_time:.2f} seconds")
            return diarization

        except Exception as e:
            logger.error(f"Error during diarization: {e}")
            return None
        finally:
            if is_temporary:
                try:
                    os.unlink(converted_audio_path)
                except Exception:
                    pass

    def format_diarization_output(self, diarization, audio_path: str) -> str:
        """Format diarization results into a readable string."""
        if diarization is None:
            return "No diarization results available."

        output_lines = [f"Speaker Diarization Results for: {os.path.basename(audio_path)}", "=" * 50]
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_time = format_timestamp(turn.start)
            end_time = format_timestamp(turn.end)
            output_lines.append(f"{start_time} --> {end_time} | {speaker}")

        return "\n".join(output_lines)

    def get_speaker_count(self, diarization) -> int:
        """Get the number of unique speakers detected."""
        if diarization is None:
            return 0
        return len(set(speaker for _, _, speaker in diarization.itertracks(yield_label=True)))
