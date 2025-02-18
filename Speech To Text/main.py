import time
from simple_transcriber import SimpleTranscriber

def handle_transcription(transcription):
    """Simply print the user query without any LLM interaction."""
    print(f"User query: {transcription}")

def main():
    # Initialize the transcriber with a callback to handle transcriptions
    transcriber = SimpleTranscriber(callback=handle_transcription)

    try:
        # Start the recording and transcription process
        transcriber.start_recording()
        print("Recording started. Speak to ask questions and stop when done.")
        
        while True:
            # Keep the main thread running
            time.sleep(1)
    except KeyboardInterrupt:
        transcriber.stop_recording()
        print("Recording stopped.")

if __name__ == "__main__":
    main()
