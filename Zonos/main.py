import requests
import json
import time
from simple_transcriber import SimpleTranscriber

# Groq Cloud API details
GROQ_API_URL = "https://api.groq.com/openai/v1"
GROQ_API_KEY = "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"

def call_groq_api(prompt):
    """Call the Groq Cloud LLM API with the generated prompt."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "prompt": prompt,
        "max_tokens": 100,  # You can adjust based on your needs
        "temperature": 0.7  # Adjust the creativity of the response
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["text"]  # Adjust based on the actual response structure
        else:
            print(f"Error calling Groq API: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error during API request: {e}")
        return None

def handle_transcription(transcription):
    """Callback function to handle new transcriptions and interact with LLM."""
    print(f"User query: {transcription}")
    
    # Generate prompt for the LLM
    prompt = f"The following is a conversation:\nUser: {transcription}\nAI:"
    
    # Get a response from the LLM
    llm_response = call_groq_api(prompt)
    
    if llm_response:
        print(f"AI response: {llm_response}")
        return llm_response
    else:
        print("No response from LLM.")
        return "I'm sorry, I didn't catch that."

def main():
    # Initialize the transcriber with a callback to handle transcriptions
    transcriber = SimpleTranscriber(callback=handle_transcription)

    try:
        # Start the recording and transcription process
        transcriber.start_recording()
        print("Recording started. Speak to ask questions and stop when done.")
        
        while True:
            # Keep the main thread running, so it keeps listening for new transcriptions
            time.sleep(1)
    except KeyboardInterrupt:
        transcriber.stop_recording()
        print("Recording stopped.")

if __name__ == "__main__":
    main()
