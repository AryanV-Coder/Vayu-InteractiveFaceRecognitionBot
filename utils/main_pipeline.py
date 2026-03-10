import asyncio
import base64
import os
import time
import pyaudio
from dotenv import load_dotenv
from sarvamai import AsyncSarvamAI, AudioOutput
from pydub import AudioSegment
from whisper_stt import stt

load_dotenv()


async def _tts_to_file(text, output_path="tts_output.mp3"):
    """Convert text to speech using Sarvam and save to MP3 file."""
    client = AsyncSarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))

    async with client.text_to_speech_streaming.connect(model="bulbul:v3") as ws:
        await ws.configure(target_language_code="hi-IN", speaker="shubh")

        await ws.convert(text)
        await ws.flush()

        chunk_count = 0
        with open(output_path, "wb") as f:
            async for message in ws:
                if isinstance(message, AudioOutput):
                    chunk_count += 1
                    audio_chunk = base64.b64decode(message.data.audio)
                    f.write(audio_chunk)
                    f.flush()

        if hasattr(ws, "_websocket") and not ws._websocket.closed:
            await ws._websocket.close()

    print(f"✅ TTS: {chunk_count} chunks saved to {output_path}")
    return output_path


def _play_audio_file(file_path):
    """Play an MP3 file through the speaker using pydub + pyaudio."""
    audio_segment = AudioSegment.from_mp3(file_path)
    raw_data = audio_segment.raw_data

    p = pyaudio.PyAudio()
    stream = p.open(
        format=p.get_format_from_width(audio_segment.sample_width),
        channels=audio_segment.channels,
        rate=audio_segment.frame_rate,
        output=True
    )
    stream.write(raw_data)
    stream.stop_stream()
    stream.close()
    p.terminate()


def speak_text(text, shared_state):
    """Convert text to speech, save to file, then play. Mutes mic during playback."""
    shared_state['bot_is_speaking'] = True
    try:
        print(f"🔊 Speaking...")
        output_path = "tts_output.mp3"
        asyncio.run(_tts_to_file(text, output_path))
        _play_audio_file(output_path)
    except Exception as e:
        print(f"❌ TTS Error: {e}")
    finally:
        time.sleep(0.2)  # Small buffer before re-enabling mic
        shared_state['bot_is_speaking'] = False


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