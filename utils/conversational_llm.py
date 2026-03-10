from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

class ConversationLLM():
    def __init__(self, name, description):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.chat_history = [
            {"role": "system", "content": f"You are a conversational chatbot. You are made by CICR. You are in a fest event. You are talking to {name}. His descpirtion is {description}. You need to talk to the person based on his description. Your first message shoud contain a greeting with the name of the person."}
        ]
    
    def send_message(self,user_text):

        self.chat_history.append({"role": "user", "content": user_text})

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # or mixtral-8x7b-32768
            messages=self.chat_history,
            stream=False,
            temperature=0.7,
            max_tokens=150
        )

        response_text = response.choices[0].message.content
        
        self.chat_history.append({"role": "assistant", "content": response_text})

        return response_text