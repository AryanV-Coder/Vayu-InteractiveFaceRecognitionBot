import asyncio
import base64
import io
from sarvamai import AsyncSarvamAI, AudioOutput
from pydub import AudioSegment
from pydub.playback import _play_with_pyaudio
import pyaudio
from dotenv import load_dotenv
import os

load_dotenv()

async def tts_stream_and_play():
    client = AsyncSarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))
    
    # Initialize PyAudio for playback
    p = pyaudio.PyAudio()
    stream = None

    async with client.text_to_speech_streaming.connect(model="bulbul:v3") as ws:
        await ws.configure(target_language_code="hi-IN", speaker="shubh")
        print("Sent configuration")

        long_text = (
            "भारत की संस्कृति विश्व की सबसे प्राचीन और समृद्ध संस्कृतियों में से एक है।"
            "यह विविधता, सहिष्णुता और परंपराओं का अद्भुत संगम है, "
            "जिसमें विभिन्न धर्म, भाषाएं, त्योहार, संगीत, नृत्य, वास्तुकला और जीवनशैली शामिल हैं।"
        )

        await ws.convert(long_text)
        print("Sent text message")

        await ws.flush()
        print("Flushed buffer")

        chunk_count = 0
        
        async for message in ws:
            if isinstance(message, AudioOutput):
                chunk_count += 1
                
                # Decode base64 to MP3 bytes
                mp3_bytes = base64.b64decode(message.data.audio)
                
                # Convert MP3 to raw PCM using pydub
                audio_segment = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
                raw_data = audio_segment.raw_data
                
                # Initialize stream on first chunk (now we know the audio params)
                if stream is None:
                    stream = p.open(
                        format=p.get_format_from_width(audio_segment.sample_width),
                        channels=audio_segment.channels,
                        rate=audio_segment.frame_rate,
                        output=True
                    )
                    print(f"Audio stream opened: {audio_segment.frame_rate}Hz, {audio_segment.channels}ch")
                
                # Play the chunk immediately
                stream.write(raw_data)
                print(f"Playing chunk {chunk_count}...")

        print(f"\n✓ Played {chunk_count} chunks")
        
        # Cleanup
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()

if __name__ == "__main__":
    asyncio.run(tts_stream_and_play())

