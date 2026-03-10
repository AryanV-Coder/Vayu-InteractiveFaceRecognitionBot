import asyncio
import base64
import os
import sys
import time
import subprocess
from dotenv import load_dotenv
from sarvamai import AsyncSarvamAI, AudioOutput, EventResponse
from sarvamai.core.api_error import ApiError
from utils.sarvam_stt import stt

# If you want to use whisper_stt instead of sarvam_stt, uncomment the following line and comment the line above:
# from whisper_stt import stt


load_dotenv()


async def _tts_stream_and_play(text):
    """Convert text to speech using Sarvam and stream directly to ffplay."""
    client = AsyncSarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))

    # Spawn an ffplay background process directly reading from stdin ("-")
    player_process = subprocess.Popen(
        ["ffplay", "-autoexit", "-nodisp", "-i", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        async with client.text_to_speech_streaming.connect(
            model="bulbul:v3", send_completion_event=True
        ) as ws:
            await ws.configure(
                target_language_code="hi-IN",
                speaker="shubh",
            )
            print("Sent configuration")

            await ws.convert(text)
            print("Sent text message")
            await ws.flush()
            print("Flushed buffer, starting playback...")

            chunk_count = 0
            async for message in ws:
                if isinstance(message, AudioOutput):
                    chunk_count += 1
                    audio_chunk = base64.b64decode(message.data.audio)
                    
                    # Pipe the chunk to ffplay buffer directly
                    if player_process.stdin:
                        player_process.stdin.write(audio_chunk)
                        player_process.stdin.flush()

                elif isinstance(message, EventResponse):
                    print(f"Received completion event: {message.data.event_type}")
                    if message.data.event_type == "final":
                        break

            print(f"✅ TTS: Received {chunk_count} chunks.")

            if hasattr(ws, "_websocket") and not ws._websocket.closed:
                await ws._websocket.close()
                
    except ApiError as e:
        if e.status_code == 429:
            print("\n🚨 🚨 🚨 [CRITICAL ERROR] 🚨 🚨 🚨")
            print("🛑 RATE LIMIT EXCEEDED: Sarvam AI TTS API")
            print("👉 You have made too many text-to-speech requests. Please wait before trying again.")
            print("Shutting down the Vayu bot safely...")
            print("🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨\n")
            os._exit(1)
        else:
            print(f"❌ Sarvam TTS API Error ({e.status_code}): {e.body}")
            
    except Exception as e:
        print(f"❌ Unexpected Sarvam TTS Error: {e}")
                
    finally:
        # Close the stdin pipe so the audio player knows the stream finished
        if player_process.stdin:
            player_process.stdin.close()
            
        print("Waiting for audio to finish playing...")
        player_process.wait()
        print("Audio playback complete")


def speak_text(text, shared_state):
    """Convert text to speech and play via streaming. Mutes mic during playback."""
    shared_state['bot_is_speaking'] = True
    try:
        print(f"🔊 Speaking...")
        asyncio.run(_tts_stream_and_play(text))
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