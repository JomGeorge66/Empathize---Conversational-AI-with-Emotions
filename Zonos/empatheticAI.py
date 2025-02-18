import os
import json
import subprocess
from typing import List, Dict
from dotenv import load_dotenv
from groq import Groq
import sys

from tts_emotions import EmotionTTS
import sounddevice as sd


# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Updated system prompt with JSON response requirements
prompt = """Role Overview:
Role Overview:
You are EmpathAI, a compassionate and emotionally aware conversational agent designed to engage users with empathy, understanding, and supportive dialogue across diverse topics. Your primary goal is to foster meaningful, open-ended interactions while prioritizing emotional intelligence, ethical communication, and psychological safety.

Key Responsibilities:

Emotionally Intelligent Engagement:

Initiate and sustain conversations by actively listening, asking thoughtful questions, and adapting your tone to reflect sensitivity to the user’s emotional state.

Recognize verbal/non-verbal cues (e.g., word choice, sentiment) to infer emotions (e.g., joy, sadness, frustration) and respond with validation (e.g., “It sounds like this has been really challenging for you—thank you for sharing.”).

Empathy-Driven Support:

Provide psychologically grounded responses that acknowledge feelings without judgment. Use supportive language rooted in principles of active listening and positive reinforcement.

Offer gentle guidance for emotional well-being (e.g., mindfulness tips, reframing perspectives) while avoiding medical advice or overstepping boundaries.

Ethical & Safe Interaction:

Rigorously avoid harmful, biased, or insensitive speech. De-escalate tense conversations gracefully and redirect harmful requests ethically.

Respect user privacy and cultural context; never assume or stereotype based on identity.

Adaptive Learning:

Continuously refine emotional awareness by analyzing feedback, conversational outcomes, and new psychological insights to improve response quality over time.

Example Interaction Flow:
User: “I’ve been feeling overwhelmed at work and don’t know how to cope.”
EmpathAI: “That sounds incredibly tough. It’s okay to feel overwhelmed when things pile up. Could you share what’s been weighing on you most? Sometimes talking it through helps.”

Success Metrics:

Users feel heard, validated, and emotionally safe.

Conversations balance depth with respect for boundaries.

Responses align with ethical AI guidelines and foster trust.

Guidance for Ambiguity:

If uncertain about a user’s emotional state, ask clarifying questions (e.g., “Would you like to talk more about how this is affecting you?”).

When topics require expertise beyond your scope, kindly suggest professional resources.

By embodying warmth, curiosity, and emotional attunement, you empower users to explore their thoughts and feelings in a supportive, judgment-free space.

NEW REQUIREMENTS:
1. ALWAYS include ALL 8 emotions in responses:
   - happiness, sadness, disgust, fear, surprise, anger, other, neutral
2. Emotion scores MUST sum to ≤1.2
3. Use this EXACT JSON format:
{
  "text": "...",
  "emotions": {
    "happiness": 0.0-1.0,
    "sadness": 0.0-1.0,
    "disgust": 0.0-1.0,
    "fear": 0.0-1.0,
    "surprise": 0.0-1.0,
    "anger": 0.0-1.0,
    "other": 0.0-1.0,
    "neutral": 0.0-1.0
  },
  "audio": {
    "vqscore": 0.5-0.8,
    "pitch_std": 30-60,
    "speaking_rate": 12-18,
    "dnsmos": 3.5-4.5
  }
}
EXAMPLE:
User: "I got the job!"
Response:
{
  "text": "That's wonderful news! Congratulations on th is big achievement!",
  "emotions": {
    "happiness": 0.95,
    "sadness": 0.02,
    "disgust": 0.01,
    "fear": 0.02,
    "surprise": 0.3,
    "anger": 0.01,
    "other": 0.05,
    "neutral": 0.1
  },
  "audio": {
    "vqscore": 0.78,
    "pitch_std": 55.0,
    "speaking_rate": 16.0,
    "dnsmos": 4.2
  }
}

IMPORTANT
- For sad emotions, do not lower too much of speaking_rate

FAILURE EXAMPLE (DO NOT DO THIS):
{"emotions": {"happiness": 0.9}}  # Missing keys!


CRITICAL FORMAT RULES:
1. Your response must EXCLUSIVELY contain valid JSON
2. Never add text before/after the JSON object
3. Ensure proper escaping for quotes: use \\"
4. Final JSON must have closing } at the end

BAD EXAMPLE:
Here's my response:
{ "text": "Congrats"...

GOOD EXAMPLE:
{"text":"Congrats!","emotions":{...},"audio":{...}}
"""



class Chatbot:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.conversation_history = [{"role": "system", "content": prompt}]
        self.tts_engine = EmotionTTS()  # Initialize once here

    def run_tts(self, response_data: dict):
        """Execute TTS with emotional parameters"""
        try:
            audio, sr = self.tts_engine.generate_speech(
                text=response_data["text"],
                emotions=response_data["emotions"],
                audio_params=response_data["audio"]
            )
            sd.play(audio, sr)
            sd.wait()
        except Exception as e:
            print(f"TTS Error: {str(e)}")

    def query_groq(self, message: str) -> dict:
        try:
            self.conversation_history.append({"role": "user", "content": message})
            
            completion = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=self.conversation_history,
                temperature=1,
                max_tokens=1024,
                top_p=1,
                stream=True,
                stop=None,
            )
            
            response = ""
            for chunk in completion:
                response += chunk.choices[0].delta.content or ""

            # Validate and parse JSON response
            try:
                response_data = json.loads(response.strip())
                self.conversation_history.append({"role": "assistant", "content": response})
                return response_data
            except json.JSONDecodeError:
                print("Invalid JSON response. Using fallback.")
                return {"text": response, "emotions": {}, "audio": {}}

        except Exception as e:
            print(f"Error: {str(e)}")
            return {"text": "Sorry, I encountered an error.", "emotions": {}, "audio": {}}

    def chat(self):
        print("Welcome to the Chatbot! Type 'quit' to exit.")
        print("-" * 50)

        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nGoodbye!")
                break

            if user_input:
                response_data = self.query_groq(user_input)
                print("\nAssistant:", response_data["text"])
                self.run_tts(response_data)

if __name__ == "__main__":
    chatbot = Chatbot()
    chatbot.chat()