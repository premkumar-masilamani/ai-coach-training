# AI-Powered Transcription Service

This application provides an automated transcription service that leverages AI to convert audio files into text. It is designed to be run from the command line and can process multiple files in a single batch.

## Features

*   **Batch Processing:** Transcribe entire directories of audio files recursively.
*   **Efficient:** Skips audio files that have already been transcribed.
*   **High-Quality Transcription:** Powered by an optimized version of OpenAI's Whisper model ([faster-whisper](https://github.com/guillaumekln/faster-whisper)).
*   **Local Caching:** The transcription model is downloaded and cached locally for faster subsequent runs.

## Prerequisites

*   Python 3
*   `pipenv`

## Setup

1.  **Install Dependencies:** Create a virtual environment and install the required Python packages.

    ```bash
    make setup
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

## How It Works

*   **Transcription Model:** The application uses the `guillaumekln/faster-whisper-medium` model from the Hugging Face Hub. This is a CTranslate2 implementation of OpenAI's Whisper model, which is up to 4 times faster than the original implementation.
*   **Model Caching:** On the first run, the script will download the model (approximately 1.5GB) and store it in the `models/guillaumekln/faster-whisper-medium/` directory. This may take some time depending on your internet connection. Subsequent runs will use the cached model.
*   **Output:** For each audio file (e.g., `meeting.mp3`), a corresponding text file (`meeting.txt`) will be created in the same location.