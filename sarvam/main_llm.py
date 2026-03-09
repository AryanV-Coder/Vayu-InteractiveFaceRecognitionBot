from dotenv import load_dotenv
load_dotenv()
# Example 2: Multi-turn example — maintaining conversation context
from sarvamai import SarvamAI
import os

client = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))

for chunk in client.chat.completions(
    model="sarvam-30b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the two main styles?"}
    ],
    stream=True,
):
    if chunk.choices:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)