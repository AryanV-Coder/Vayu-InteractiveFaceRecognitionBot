import os
import sys
from sarvamai import SarvamAI
from sarvamai.core.api_error import ApiError
from dotenv import load_dotenv

load_dotenv()

def stt(audio_path):
    """
    Transcribe audio from a file path using Sarvam AI's Saaras model.
    Matches the signature of the old whisper_stt.py for easy replacement.
    """
    client = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))

    try:
        response = client.speech_to_text.transcribe(
            file=open(audio_path, "rb"),
            model="saaras:v3",
            mode="transcribe",
        )
        
        # Determine language (optional debugging)
        if hasattr(response, 'language_code') and response.language_code:
            lang = response.language_code
        else:
            lang = "unknown"
            
        print(f"\n[Sarvam STT] Detected language: {lang}")
        
        return response.transcript
        
    except ApiError as e:
        if e.status_code == 429:
            print("\n🚨 🚨 🚨 [CRITICAL ERROR] 🚨 🚨 🚨")
            print("🛑 RATE LIMIT EXCEEDED: Sarvam AI STT API")
            print("👉 You have made too many speech-to-text requests. Please wait before trying again.")
            print("Shutting down the Vayu bot safely...")
            print("🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨\n")
            
            # Use os._exit to immediately kill the process and all threads
            os._exit(1)
        else:
            print(f"❌ Sarvam STT API Error ({e.status_code}): {e.body}")
            return ""
            
    except Exception as e:
        print(f"❌ Unexpected Sarvam STT Error: {e}")
        return ""
