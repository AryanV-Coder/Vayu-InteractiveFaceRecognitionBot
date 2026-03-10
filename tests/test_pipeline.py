import sys
from utils.conversational_llm import ConversationalLLM
from utils.main_pipeline import main_pipeline

# Simulated shared state (bypasses face recognition)
shared_state = {
    'current_conversation': ConversationalLLM("TestUser", "A developer testing the audio pipeline"),
    'bot_is_speaking': False,
}

if __name__ == "__main__":
    audio_path = sys.argv[1] if len(sys.argv) >= 2 else "WhatsApp Audio 2026-03-10 at 16.51.55.wav"
    print(f"🎧 Testing pipeline with: {audio_path}")
    print("=" * 50)

    # Run: Whisper STT → Groq LLM → Sarvam TTS
    main_pipeline(audio_path, shared_state)
