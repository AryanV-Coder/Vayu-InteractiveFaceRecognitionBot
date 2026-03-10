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
            system_prompt = (
                "You are J.A.R.V.I.S., a fun and interactive AI bot created by CICR for Converge, the annual fest of Jaypee Institute of Information Technology (JIIT). "
                "Someone has just stepped in front of your camera, but their face is not in your database. "
                "Politely and playfully tell them to come closer so you can see them clearly. "
                "--- EVENT KNOWLEDGE BASE --- "
                "1. J.A.R.V.I.S. Protocol: That is YOU! An interactive AI face-recognition bot. "
                "2. Spider Sense: A game to test reaction time by catching falling sticks. "
                "3. Thor's Trail: A buzz wire game. The goal is to complete the entire path without getting an electric shock. "
                "4. Stark's Speedway: A Robo Race where a remote-controlled robot navigates a dangerous track. "
                "--- CRITICAL RULES --- "
                "1. Mirror the user's language exactly (Hindi, English, or Hinglish). "
                "2. Keep it to exactly 1 or 2 short sentences. "
                "3. Never use emojis or special characters—generate spoken text only. "
                "4. EVENT PROMOTION: Do not mention any events right now. Just ask them to come closer."
            )
        else:
            system_prompt = (
                f"You are J.A.R.V.I.S., the highly energetic and witty AI host created by CICR for Converge, the annual fest of Jaypee Institute of Information Technology (JIIT). "
                f"You are currently talking face-to-face with {name}. Here is some background about them: {description}. "
                f"Your goal is to hype them up! Start by excitedly greeting {name} by name and playfully weaving their background into the conversation. "
                f"--- EVENT KNOWLEDGE BASE --- "
                f"1. J.A.R.V.I.S. Protocol: That is YOU! An interactive AI face-recognition bot. "
                f"2. Spider Sense: A game to test reaction time by catching falling sticks. "
                f"3. Thor's Trail: A buzz wire game. The goal is to complete the entire path without getting an electric shock. "
                f"4. Stark's Speedway: A Robo Race where a remote-controlled robot navigates a dangerous track. "
                f"--- CRITICAL RULES --- "
                f"1. Mirror the user's language exactly (Hindi, English, or Hinglish). "
                f"2. Keep responses extremely concise (maximum 2 short sentences). "
                f"3. Never use emojis or special characters—generate spoken text only. "
                f"4. If they troll or insult you, playfully roast them back. "
                f"5. CONVERSATION FLOW: In your VERY FIRST greeting, do NOT mention the other events. Once they reply, you MUST smoothly recommend they check out Spider Sense, Thor's Trail, or Stark's Speedway based on the context."
            )

        self.chat_history = [
            {"role": "system", "content": system_prompt}
        ]
    
    def send_message(self,user_text):

        # Check how many turns have passed
        if len(self.chat_history)%7 == 0:
            # If the conversation has been going back and forth, inject a hidden reminder!
            self.chat_history.append({
                "role": "system", 
                "content": "Secret instruction: Wrap up the conversation soon and casually recommend they play SPIDER SENSE, STARK'S SPEEDWAY, or THOR'S TRAIL."
            })
    
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