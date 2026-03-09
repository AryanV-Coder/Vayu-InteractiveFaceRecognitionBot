import asyncio
import base64
from sarvamai import AsyncSarvamAI, AudioOutput
from dotenv import load_dotenv
import os

async def tts_stream():
    client = AsyncSarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))

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
        with open("output.mp3", "wb") as f:
            async for message in ws:
                if isinstance(message, AudioOutput):
                    chunk_count += 1
                    audio_chunk = base64.b64decode(message.data.audio)
                    f.write(audio_chunk)
                    f.flush()

        print(f"All {chunk_count} chunks saved to output.mp3")
        print("Audio generation complete")


        if hasattr(ws, "_websocket") and not ws._websocket.closed:
            await ws._websocket.close()
            print("WebSocket connection closed.")


if __name__ == "__main__":
    asyncio.run(tts_stream())

# --- Notebook/Colab usage ---
# await tts_stream()
