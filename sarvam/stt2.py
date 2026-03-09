from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

from sarvamai import SarvamAI

client = SarvamAI(
    api_subscription_key=os.getenv("SARVAM_API_KEY"),
)

# Transcribe mode (default)
response = client.speech_to_text.transcribe(
    file=open("sarvam/audio/t.wav", "rb"),
    model="saaras:v3",
    mode="transcribe"  # or "translate", "verbatim", "translit", "codemix"
)

print(response)

