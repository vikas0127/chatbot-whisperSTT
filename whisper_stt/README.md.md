# Whisper Speech-to-Text (STT) Service

This is a self-contained Python backend service using FastAPI to provide high-quality speech-to-text transcription via the local Whisper model.

# System Prerequisites

This service requires **FFmpeg** to be installed on the machine where it is deployed.
Installation Guide: [https://ffmpeg.org/download.html]
 
 How to Run This Service Locally

1 Clone this repository.

2 Create and activate a Python virtual environment:
    ```bash
    # Create the environment
    python -m venv venv
    
    # Activate on Windows (PowerShell)
    .\venv\Scripts\Activate.ps1
    

3 Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```

4 Run the FastAPI server:
    ```bash
    uvicorn main:app --reload
    ```
    - The server will start on `http://127.0.0.1:8000`.
    - The first time it runs, it will download the Whisper `base` model (~142MB). This is a one-time download.

---

API Documentation

# Endpoint: `POST /transcribe`

Accepts an audio file and returns the transcribed text.

- URL: `http://127.0.0.1:8000/transcribe`
- Method: `POST`
- Body: `multipart/form-data` with a single field named `file` containing the audio data.

# Example `curl` Request:
```bash
curl.exe -X POST -F "file=@test_audio.m4a" 
http://127.0.0.1:8000/transcribe