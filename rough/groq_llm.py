from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

chat_history = [
    {"role": "system", "content": "Talk in a roasting way"},
    {"role": "user", "content": "Hello!"}
]

# response = client.chat.completions.create(
#     model="llama-3.3-70b-versatile",  # or mixtral-8x7b-32768
#     messages=chat_history,
#     stream=True,
#     temperature=0.7,
#     max_tokens=150
# )

# for chunk in response:
#     if chunk.choices[0].delta.content:
#         print(chunk.choices[0].delta.content, end="")

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",  # or mixtral-8x7b-32768
    messages=chat_history,
    stream=False,
    temperature=0.7,
    max_tokens=150
)

# Print the completion returned by the LLM.
print(response.choices[0].message.content)