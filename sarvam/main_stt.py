from sarvamai import SarvamAI
from dotenv import load_dotenv
import os

load_dotenv()

client = SarvamAI(
    api_subscription_key=os.getenv("SARVAM_API_KEY"),
)

response = client.speech_to_text.transcribe(
    file=open("WhatsApp Audio 2026-03-11 at 02.23.26.mp3", "rb"),
    model="saaras:v3",
    mode="transcribe",
)

print(response)
