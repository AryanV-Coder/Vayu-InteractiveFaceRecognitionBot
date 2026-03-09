from sarvamai import SarvamAI
from dotenv import load_dotenv
import os

client = SarvamAI(
    api_subscription_key=os.getenv("SARVAM_API_KEY"),
)

response = client.speech_to_text.transcribe(
    file=open("audio.wav", "rb"),
    model="saaras:v3",
    mode="transcribe",
    language_code="hi-IN"
)

print(response)
