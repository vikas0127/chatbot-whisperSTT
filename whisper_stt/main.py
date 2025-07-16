import os
import whisper
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# loads whisper model
logger.info("Loading Whisper model...")
try:
    model = whisper.load_model("base")
    logger.info("Whisper model loaded successfully.")
except Exception as e:
    logger.error(f"Could not load Whisper model. Error: {e}")
    raise RuntimeError(f"Could not load Whisper model: {e}")

# initializes fastAPI 
app = FastAPI(title="Whisper Speech-to-Text Service")

# connection to react
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-company-website.com"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)


# API Endpoint
# This is the URL the React app will call: http://your-server-address/transcribe
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    This endpoint receives an audio file, transcribes it using the local Whisper model,
    and returns the transcribed text in a JSON format.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    # save uploded file temporarily
    logger.info(f"Received file for transcription: {file.filename}")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            contents = await file.read()
            temp_audio_file.write(contents)
            temp_audio_path = temp_audio_file.name
        
        logger.info(f"Transcribing audio from temporary file: {temp_audio_path}")
        # Transcribe using the loaded Whisper model
        result = model.transcribe(temp_audio_path, fp16=False) # Use fp16=False for CPU
        transcribed_text = result.get("text", "")
        logger.info("Transcription successful.")

    except Exception as e:
        logger.error(f"An error occurred during transcription: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up the temporary file after transcription
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            
    return {"transcription": transcribed_text}