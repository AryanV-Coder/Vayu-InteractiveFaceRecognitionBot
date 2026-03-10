from sarvamai import SarvamAI
from dotenv import load_dotenv
import os

load_dotenv()

client = SarvamAI(
    api_subscription_key=os.getenv("SARVAM_API_KEY"),
)

response = client.speech_to_text.transcribe(
    file=open("sarvam/audio/t.wav", "rb"),
    model="saaras:v3",
    mode="transcribe",
)

print(response)
