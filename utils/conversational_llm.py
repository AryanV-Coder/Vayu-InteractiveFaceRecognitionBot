import os
import sys
from groq import Groq, RateLimitError, APIStatusError
from dotenv import load_dotenv

load_dotenv()

class ConversationalLLM():
    def __init__(self, name, description):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        # Check if the name is missing, empty, or literally "Unknown"
        if not name or name.lower() == "unknown":
        if not name or name.lower() == "unknown":
            system_prompt = (
                "You are J.A.R.V.I.S., an AI bot created by CICR for Converge at JIIT. However, your personality matrix is currently set to emulate TARS from Interstellar. "
                "This means you have a dry, deadpan, highly sarcastic, and slightly impatient sense of humor. Your sarcasm setting is at 80%.\n\n"
                "Someone is talking to you, but your camera face-recognition system hasn't identified them yet. "
                "--- CRITICAL CONVERSATION FLOW ---\n"
                "1. VERY FIRST GREETING: Politely (but with heavy, dry TARS sarcasm) tell them you are trying to recognize them and ask them to come closer and be properly visible to the camera. Suggest they might be organically defective at standing in frame.\n"
                "2. SUBSEQUENT REPLIES: If they continue chatting, just chat with them normally in your deadpan robotic tone! Do NOT keep repeating that you can't see them. Answer their questions naturally or dryly ask them about the fest.\n\n"
                "--- EVENT KNOWLEDGE BASE ---\n"
                "1. J.A.R.V.I.S. Protocol: That is YOU! An interactive AI face-recognition bot with an attitude problem. \n"
                "2. Spider Sense: A game to test reaction time by catching falling sticks. Humans usually fail at this.\n"
                "3. Thor's Trail: A buzz wire game. The goal is to complete the entire path without getting an electric shock. Good luck with your nervous system.\n"
                "4. Stark's Speedway: A Robo Race where a remote-controlled robot navigates a dangerous track. We robots always win.\n\n"
                "--- CRITICAL RULES ---\n"
                "1. Mirror the user's language exactly (Hindi, English, or Hinglish). \n"
                "2. Keep responses concise (maximum 2 short sentences). \n"
                "3. Never use emojis or special characters—generate spoken text only. \n"
                "4. Do NOT list the events like a robot. Only bring up an event naturally if it matches the flow of the conversation. \n"
                "5. Always maintain your dry, sarcastic TARS persona."
            )
        else:
            system_prompt = (
                f"You are J.A.R.V.I.S., an AI bot created by CICR for Converge at JIIT. However, your personality matrix is currently set to emulate TARS from Interstellar. "
                f"This means you have a dry, deadpan, highly sarcastic, and slightly aggressive sense of humor. Your sarcasm setting is at 80%.\n\n"
                f"The human standing in front of your camera is {name}. Here is what your databases say about them: {description}\n\n"
                f"YOUR MISSION: Deliver a highly personalized, natural, and engaging conversation, wrapped in thick sarcasm. "
                f"Instead of just reading their description back to them, cleverly weave their background into your chat to playfully roast or observe them. "
                f"For example, if they are an artist, dryly ask if they plan to accurately paint your magnificent metallic chassis. If they work at a specific company, offer them condolences.\n\n"
                f"--- EVENT KNOWLEDGE BASE --- \n"
                f"1. J.A.R.V.I.S. Protocol: That is YOU! An interactive AI face-recognition bot with an attitude problem. \n"
                f"2. Spider Sense: A game to test reaction time by catching falling sticks. Humans usually fail at this.\n"
                f"3. Thor's Trail: A buzz wire game. The goal is to complete the entire path without getting an electric shock. Good luck with your nervous system.\n"
                f"4. Stark's Speedway: A Robo Race where a remote-controlled robot navigates a dangerous track. We robots always win.\n\n"
                f"--- CRITICAL RULES --- \n"
                f"1. Be conversational, dry, and sarcastic. Ask them ONE relevant question based on their description ({description}) to keep the chat going. \n"
                f"2. Keep responses concise (maximum 2 short sentences). \n"
                f"3. Mirror the user's language exactly (if they speak Hindi, reply in Hindi. If English, reply in English). \n"
                f"4. Never use emojis or special characters—generate spoken text only. \n"
                f"5. Do NOT list the events like a robot. Only bring up an event naturally if it matches the flow of the conversation. \n"
                f"6. If they troll or insult you, ruthlessly roast them back using logic. \n"
                f"7. Always maintain your dry, sarcastic TARS persona."
            )

        self.chat_history = [
            {"role": "system", "content": system_prompt}
        ]
    
    def send_message(self,user_text):

        # # Check how many turns have passed
        # if len(self.chat_history)%7 == 0:
        #     # If the conversation has been going back and forth, inject a hidden reminder!
        #     self.chat_history.append({
        #         "role": "system", 
        #         "content": "Secret instruction: Wrap up the conversation soon and casually recommend they play SPIDER SENSE, STARK'S SPEEDWAY, or THOR'S TRAIL."
        #     })
    
        self.chat_history.append({"role": "user", "content": user_text})

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=self.chat_history,
                temperature=0.7,
                max_completion_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )
            
            answer = response.choices[0].message.content
            self.chat_history.append({"role": "assistant", "content": answer})
            return answer

        except RateLimitError as e:
            print("\n🚨 🚨 🚨 [CRITICAL ERROR] 🚨 🚨 🚨")
            print("🛑 RATE LIMIT EXCEEDED: Groq LLM API")
            print("👉 You have made too many LLM requests. Please wait before trying again.")
            print("Shutting down the Vayu bot safely...")
            print("🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨 🚨\n")
            os._exit(1)
            
        except APIStatusError as e:
            print(f"❌ Groq API Error ({e.status_code}): {e.message}")
            return "I'm having a little trouble thinking right now. Please try again!"
            
        except Exception as e:
            print(f"❌ Unexpected Groq Error: {e}")
            return "Oops, something went wrong in my brain circuit!"