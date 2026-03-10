import asyncio
import base64
import os
import subprocess
import time
from dotenv import load_dotenv
from sarvamai import AsyncSarvamAI, AudioOutput, EventResponse

load_dotenv()

async def tts_stream():
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

            long_text = (
                "भारत की संस्कृति विश्व की सबसे प्राचीन और समृद्ध संस्कृतियों में से एक है।"
                "यह विविधता, सहिष्णुता और परंपराओं का अद्भुत संगम है, "
                "जिसमें विभिन्न धर्म, भाषाएं, त्योहार, संगीत, नृत्य, वास्तुकला और जीवनशैली शामिल हैं।"
            )

            start_time = time.time()
            await ws.convert(long_text)
            print("Sent text message")
            await ws.flush()
            print("Flushed buffer, starting playback...")

            chunk_count = 0
            async for message in ws:
                if isinstance(message, AudioOutput):
                    if chunk_count == 0:
                        ttfb = time.time() - start_time
                        print(f"\n[Time to First Audio Byte: {ttfb:.3f} seconds]")
                        print("Receiving chunks: ", end="", flush=True)
                    
                    chunk_count += 1
                    print(".", end="", flush=True)
                    
                    audio_chunk = base64.b64decode(message.data.audio)
                    
                    # Pipe the chunk to ffplay buffer directly
                    if player_process.stdin:
                        player_process.stdin.write(audio_chunk)
                        player_process.stdin.flush()

                elif isinstance(message, EventResponse):
                    print(f"\nReceived completion event: {message.data.event_type}")
                    if message.data.event_type == "final":
                        break

            print(f"Received total {chunk_count} chunks.")

            if hasattr(ws, "_websocket") and not ws._websocket.closed:
                await ws._websocket.close()
                
    finally:
        # Close the stdin pipe so the audio player knows the stream finished
        if player_process.stdin:
            player_process.stdin.close()
            
        print("Waiting for audio to finish playing...")
        player_process.wait()
        print("Audio playback complete")


if __name__ == "__main__":
    asyncio.run(tts_stream())
