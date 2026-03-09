import asyncio
import base64
import os
import pyaudio
from dotenv import load_dotenv
from sarvamai import AsyncSarvamAI, AudioOutput

load_dotenv()

# --- HARDWARE CONFIGURATION ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

audio = pyaudio.PyAudio()

# Initialize Sarvam Async Client
client = AsyncSarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))

# --- ASYNC QUEUES & STATE ---
llm_queue = asyncio.Queue()
tts_queue = asyncio.Queue()
speaker_queue = asyncio.Queue()

# State flag to prevent the bot from hearing its own voice
bot_is_speaking = False 

# Conversation Memory
chat_history = [
    {"role": "system", "content": "You are a witty, energetic interactive bot at a college fest. Keep your responses to 1 or 2 short sentences. Speak in a mix of Hindi and English naturally."}
]

async def stt_worker():
    """Worker 1: The Ear (Mic -> Sarvam STT)"""
    global bot_is_speaking
    
    # Open microphone stream
    mic_stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    async with client.speech_to_text_streaming.connect(
        model="saaras:v3",
        mode="transcribe",
        language_code="hi-IN", # Supports Hinglish automatically
        high_vad_sensitivity=True,
        vad_signals=True 
    ) as ws:
        print("🎙️ Bot is listening...")
        
        # Background task to continuously send mic audio
        async def send_audio():
            while True:
                # Read from mic (using to_thread to prevent blocking the async loop)
                data = await asyncio.to_thread(mic_stream.read, CHUNK, exception_on_overflow=False)
                
                # Only send audio to Sarvam if the bot is NOT currently speaking
                if not bot_is_speaking:
                    base64_audio = base64.b64encode(data).decode("utf-8")
                    await ws.transcribe(audio=base64_audio, encoding="audio/wav", sample_rate=RATE)
                await asyncio.sleep(0.01)

        asyncio.create_task(send_audio())
        
        # Listen for transcription results
        async for message in ws:
            if message.get("type") == "transcript" and not bot_is_speaking:
                user_text = message.get("text").strip()
                if user_text:
                    print(f"\n👤 You: {user_text}")
                    await llm_queue.put(user_text)

async def llm_worker():
    """Worker 2: The Brain (Transcript -> Sarvam LLM -> Text Chunks)"""
    while True:
        user_text = await llm_queue.get()
        
        # Append user message to history
        chat_history.append({"role": "user", "content": user_text})
        
        print("🤖 Thinking...")
        full_response = ""
        chunk_buffer = ""
        
        # Call Sarvam's chat completion with streaming
        # Note: Depending on Sarvam's exact async SDK implementation, 
        # this awaits the stream generator.
        response_stream = await client.chat.completions.create(
            model="sarvam-30b",
            messages=chat_history,
            stream=True
        )
        
        async for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_response += token
                chunk_buffer += token
                
                # Check for punctuation to trigger a TTS chunk
                if any(p in token for p in [".", ",", "?", "!", "।"]): # Included Hindi Purna Viram
                    await tts_queue.put(chunk_buffer.strip())
                    chunk_buffer = "" 
        
        # Flush any remaining text
        if chunk_buffer.strip():
            await tts_queue.put(chunk_buffer.strip())
            
        # Append the assistant's complete response to the history
        chat_history.append({"role": "assistant", "content": full_response})
        print(f"\n🤖 Bot: {full_response}")

async def tts_worker():
    """Worker 3: The Voice Engine (Text Chunks -> Sarvam TTS -> Audio Bytes)"""
    async with client.text_to_speech_streaming.connect(
        model="bulbul:v3", 
        send_completion_event=True
    ) as ws:
        # Configure the voice once
        await ws.configure(target_language_code="hi-IN", speaker="shubh")
        
        # Background task to receive audio
        async def receive_audio():
            async for message in ws:
                if isinstance(message, AudioOutput):
                    audio_bytes = base64.b64decode(message.data.audio)
                    await speaker_queue.put(audio_bytes)
                    
        asyncio.create_task(receive_audio())
        
        while True:
            text_chunk = await tts_queue.get()
            await ws.convert(text_chunk)
            await ws.flush() 

async def speaker_worker():
    """Worker 4: The Mouth (Audio Bytes -> PyAudio Speaker)"""
    global bot_is_speaking
    
    # Open speaker stream
    speaker_stream = audio.open(format=FORMAT, channels=CHANNELS,
                                rate=RATE, output=True)
    
    while True:
        audio_bytes = await speaker_queue.get()
        
        # Lock the microphone so the bot doesn't hear itself
        bot_is_speaking = True 
        
        # Play the audio chunk
        await asyncio.to_thread(speaker_stream.write, audio_bytes)
        
        # If the queue is empty, the bot has finished its sentence. Unlock the mic.
        if speaker_queue.empty():
            bot_is_speaking = False

async def main():
    try:
        await asyncio.gather(
            stt_worker(),
            llm_worker(),
            tts_worker(),
            speaker_worker()
        )
    except KeyboardInterrupt:
        print("\nShutting down bot...")
    finally:
        audio.terminate()

if __name__ == "__main__":
    asyncio.run(main())