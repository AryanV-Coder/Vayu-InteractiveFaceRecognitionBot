import os
from dotenv import load_dotenv
from utils.sarvam_tts import speak_text
from utils.sarvam_stt import stt

# If you want to use whisper_stt instead of sarvam_stt, uncomment the following line and comment the line above:
# from utils.whisper_stt import stt


load_dotenv()


def main_pipeline(audio_path, shared_state):
    """
    Process recorded audio: STT (Whisper) → LLM (Groq) → TTS (Sarvam)

    Args:
        audio_path: Path to the recorded audio WAV file
        shared_state: Shared state dict with 'current_conversation' and 'bot_is_speaking'
    """
    conversation = shared_state.get('current_conversation')

    if not conversation:
        print("⚠️  No conversation context - waiting for face recognition...")
        return

    # Step 1: STT - Whisper
    try:
        user_text = stt(audio_path)
    except Exception as e:
        print(f"❌ STT Error: {e}")
        return

    if not user_text or not user_text.strip():
        print("⚠️  No transcription detected")
        return

    print(f"👤 User: {user_text}")

    # Step 2: LLM - Groq
    try:
        print("🤖 Thinking...")
        response = conversation.send_message(user_text)
        print(f"🤖 Bot: {response}")
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return

    # Step 3: TTS - Sarvam
    speak_text(response, shared_state)