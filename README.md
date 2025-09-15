# AI-Powered Audio Processing Suite

This application provides automated audio processing services including transcription and speaker diarization. It leverages AI to convert audio files into text and identify different speakers. It is designed to be run from the command line and can process multiple files in a single batch.

## Features

### Transcription Service
*   **Batch Processing:** Transcribe entire directories of audio files recursively.
*   **Efficient:** Skips audio files that have already been transcribed.
*   **High-Quality Transcription:** Powered by an optimized version of OpenAI's Whisper model ([faster-whisper](https://github.com/guillaumekln/faster-whisper)).
*   **Local Caching:** The transcription model is downloaded and cached locally for faster subsequent runs.

### Speaker Diarization Service
*   **Speaker Identification:** Identify and separate different speakers in audio recordings.
*   **Offline Operation:** Works completely offline after initial model download.
*   **Multiple Output Formats:** Generate timestamped speaker segments.
*   **Batch Processing:** Process multiple files with speaker diarization.
*   **Local Model Caching:** Pyannote models are downloaded once and cached locally.

## Prerequisites

*   Python 3.13 (or compatible version)
*   `pipenv`
*   **For Speaker Diarization:** Hugging Face account and token (free)

## Setup

1.  **Install Dependencies:** Create a virtual environment and install the required Python packages.

    ```bash
    make setup
    ```

2.  **Setup Hugging Face Token (for diarization):** If you plan to use speaker diarization:

    ```bash
    python setup_hf_token.py
    ```

    This will guide you through setting up your Hugging Face token for downloading the speaker diarization model.

3.  **Verify Setup (optional):**

    ```bash
    python setup_hf_token.py verify
    ```

## Usage

1.  **Configure Input Directory:**
    Open the `Makefile` and modify the `INPUT_DIR` variable to point to the absolute path of the directory containing your audio files.

    ```makefile
    INPUT_DIR = "/path/to/your/audio/files"
    ```

2.  **Run Transcription:**
    Execute the following command to start the transcription process:

    ```bash
    make run
    ```

    The script will scan the `INPUT_DIR` for supported audio files (`.mp3`, `.wav`, `.m4a`, `.flac`, `.aac`, `.ogg`) and generate a text file (`.txt`) for each one in the same directory.

### Direct Usage (without Make)

If you prefer not to use `make`, you can run the script directly:

1.  **Activate Virtual Environment:**
    ```bash
    pipenv shell
    ```

2.  **Run the script:**
    ```bash
    python3 transcribe.py -i /path/to/your/audio/files
    ```
    You can use the `-v` flag for more detailed logging output.

## Speaker Diarization Usage

The speaker diarization service can identify different speakers in audio recordings.

### Single File Processing

```bash
python example_diarization.py
```

Edit the `audio_file` path in the script to point to your audio file.

### Batch Processing

```bash
python example_diarization.py batch /path/to/audio/directory
```

This will process all audio files in the directory and create `.diarization.txt` files with speaker timestamps.

### Programmatic Usage

```python
from diarization_service import PyannoteDiarizer

# Initialize (downloads model on first run, then works offline)
diarizer = PyannoteDiarizer()

# Process audio file
diarization = diarizer.diarize("path/to/audio.wav")

# Get formatted output
output = diarizer.format_diarization_output(diarization, "audio.wav")
print(output)

# Get speaker count
speaker_count = diarizer.get_speaker_count(diarization)
print(f"Found {speaker_count} speakers")
```

## How It Works

### Transcription Service
*   **Transcription Model:** Uses the `guillaumekln/faster-whisper-medium` model from Hugging Face Hub. This is a CTranslate2 implementation of OpenAI's Whisper model, up to 4x faster than the original.
*   **Model Caching:** Downloads the model (~1.5GB) on first run and stores in `models/guillaumekln/faster-whisper-medium/`. Subsequent runs use the cached model.
*   **Output:** Creates `.txt` files with transcriptions for each audio file.

### Speaker Diarization Service
*   **Diarization Model:** Uses the `pyannote/speaker-diarization-3.1` model for state-of-the-art speaker separation.
*   **Offline Operation:** After initial download (~1GB), works completely offline without internet connection.
*   **Model Caching:** Downloads and caches the model locally in `models/pyannote/speaker-diarization-3.1/`.
*   **Output:** Creates `.diarization.txt` files with timestamped speaker segments.
*   **Authentication:** Requires Hugging Face token for initial download (accepts model license agreement).
