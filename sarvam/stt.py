import asyncio
import base64
import pyaudio
import os
from dotenv import load_dotenv
from sarvamai import AsyncSarvamAI

# Load environment variables
load_dotenv()

# Audio configuration
CHUNK = 1024  # Audio chunk size
FORMAT = pyaudio.paInt16  # 16-bit audio
CHANNELS = 1  # Mono audio
RATE = 16000  # Sample rate (16kHz for Sarvam API)


async def microphone_streaming(duration_seconds=10):
    """
    Capture audio from microphone and stream to Sarvam AI for transcription
    
    Args:
        duration_seconds: How long to record (default: 10 seconds)
    """
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise ValueError("SARVAM_API_KEY not found in environment variables")
    
    client = AsyncSarvamAI(api_subscription_key=api_key)
    
    # Initialize PyAudio
    audio = pyaudio.PyAudio()
    
    print(f"🎤 Recording for {duration_seconds} seconds...")
    print("Speak now!")
    
    # Open microphone stream
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    try:
        async with client.speech_to_text_streaming.connect(
            model="saaras:v3",
            mode="transcribe",
            language_code="en-IN",
            flush_signal=True
        ) as ws:
            
            # Calculate number of chunks to record
            total_chunks = int(RATE / CHUNK * duration_seconds)
            
            # Record and stream audio
            for i in range(total_chunks):
                # Read audio chunk from microphone
                audio_chunk = stream.read(CHUNK, exception_on_overflow=False)
                
                # Encode to base64
                audio_b64 = base64.b64encode(audio_chunk).decode("utf-8")
                
                # Send to API
                await ws.transcribe(
                    audio=audio_b64,
                    encoding="audio/wav",
                    sample_rate=RATE
                )
                
                # Progress indicator
                if (i + 1) % 16 == 0:  # Update every ~1 second
                    print(f"Recording... {(i + 1) / total_chunks * 100:.0f}%")
            
            print("🛑 Recording finished. Processing...")
            
            # Force processing of remaining audio
            await ws.flush()
            
            # Collect transcription results
            print("\n📝 Transcription:")
            print("-" * 50)
            async for message in ws:
                if hasattr(message, 'transcript'):
                    print(f"{message.transcript}")
                else:
                    print(f"{message}")
    
    finally:
        # Clean up
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print("\n✓ Audio stream closed")


async def continuous_listening():
    """
    Continuously listen and transcribe in real-time
    Press Ctrl+C to stop
    """
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise ValueError("SARVAM_API_KEY not found in environment variables")
    
    client = AsyncSarvamAI(api_subscription_key=api_key)
    audio = pyaudio.PyAudio()
    
    print("🎤 Continuous listening mode activated")
    print("Press Ctrl+C to stop")
    print("Speak into your microphone...\n")
    
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    stop_event = asyncio.Event()
    
    async def send_audio(ws):
        """Send audio chunks to the API"""
        try:
            while not stop_event.is_set():
                # Read audio chunk
                audio_chunk = stream.read(CHUNK, exception_on_overflow=False)
                audio_b64 = base64.b64encode(audio_chunk).decode("utf-8")
                
                # Send to API
                await ws.transcribe(
                    audio=audio_b64,
                    encoding="audio/wav",
                    sample_rate=RATE
                )
                
                # Small delay to prevent overwhelming the API
                await asyncio.sleep(0.01)
        except Exception as e:
            print(f"Send error: {e}")
            stop_event.set()
    
    async def receive_transcripts(ws):
        """Receive transcription results from the API"""
        try:
            async for message in ws:
                if hasattr(message, 'transcript') and message.transcript:
                    print(f"📝 {message.transcript}")
                elif hasattr(message, 'text') and message.text:
                    print(f"📝 {message.text}")
                else:
                    print(f"📝 {message}")
        except Exception as e:
            print(f"Receive error: {e}")
            stop_event.set()
    
    try:
        async with client.speech_to_text_streaming.connect(
            model="saaras:v3",
            mode="transcribe",
            language_code="en-IN",
            flush_signal=True
        ) as ws:
            
            # Run sending and receiving concurrently
            send_task = asyncio.create_task(send_audio(ws))
            receive_task = asyncio.create_task(receive_transcripts(ws))
            
            try:
                await asyncio.gather(send_task, receive_task)
            except KeyboardInterrupt:
                print("\n\n🛑 Stopping...")
                stop_event.set()
                send_task.cancel()
                receive_task.cancel()
    
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping...")
    
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print("✓ Audio stream closed")


if __name__ == "__main__":
    print("=== Sarvam AI Speech-to-Text (Microphone Input) ===\n")
    print("Choose mode:")
    print("1. Record for specific duration (default 10 seconds)")
    print("2. Continuous listening (until Ctrl+C)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "2":
        asyncio.run(continuous_listening())
    else:
        duration = input("Enter recording duration in seconds (default 10): ").strip()
        duration = int(duration) if duration.isdigit() else 10
        asyncio.run(microphone_streaming(duration_seconds=duration))
