import pyaudio
import numpy as np
import threading
import queue
import os
from faster_whisper import WhisperModel


class SimpleTranscriber:

    def __init__(self, model_size="large", device="cuda", compute_type="float16", callback=None):
        # Initialize Whisper Model
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=min(os.cpu_count(), 8)
        )

        # Audio Settings
        self.CHUNK = 1600
        self.RATE = 16000
        self.CHANNELS = 1

        # Silence Detection Settings
        self.SILENCE_THRESHOLD = 0.02  # Adjust this for sensitivity
        self.SILENCE_WINDOW = self.RATE // 2  # Duration of silence to detect (0.5 seconds)
        self.MIN_AUDIO_LENGTH = self.RATE * 2  # Minimum length of 3 seconds of audio

        # Processing Variables
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.is_recording = False

        # Initialize PyAudio
        self.p = pyaudio.PyAudio()

        # Callback function to handle transcripts
        self.callback = callback

    def start_recording(self):
        """Start recording and transcription"""
        try:
            self.stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )

            self.is_recording = True
            self.stop_event.clear()

            # Start recording thread
            threading.Thread(target=self.audio_capture_thread).start()

            # Start transcription thread
            threading.Thread(target=self.transcribe_audio).start()

        except Exception as e:
            print(f"Could not start recording: {str(e)}")

    def stop_recording(self):
        """Stop recording and transcription"""
        self.is_recording = False
        self.stop_event.set()
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

    def audio_capture_thread(self):
        """Capture audio chunks"""
        while not self.stop_event.is_set():
            try:
                audio_data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                audio_np = np.frombuffer(audio_data, dtype=np.float32)
                self.audio_queue.put(audio_np)
            except Exception as e:
                print(f"Audio capture error: {e}")
                break

    def is_significant_audio(self, audio_chunk):
        """Check if the audio chunk has significant volume"""
        energy = np.mean(np.abs(audio_chunk))
        return energy > self.SILENCE_THRESHOLD

    def transcribe_audio(self):
        """Transcribe audio"""
        audio_buffer = []
        silence_counter = 0

        while not self.stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=1)

                # Check if audio is significant
                if self.is_significant_audio(chunk):
                    silence_counter = 0  # Reset silence counter
                    audio_buffer.extend(chunk)
                else:
                    silence_counter += len(chunk)

                # If silence is detected and buffer has sufficient audio
                if silence_counter >= self.SILENCE_WINDOW and len(audio_buffer) >= self.MIN_AUDIO_LENGTH:
                    audio_np = np.array(audio_buffer)
                    segments, _ = self.model.transcribe(
                        audio_np,
                        language="en",
                        vad_filter=True,
                        beam_size=1
                    )

                    # Generate transcription
                    current_transcription = ""
                    for segment in segments:
                        text = segment.text.strip()
                        if text:
                            current_transcription += text + " "

                    # Trigger callback with the new transcript
                    if current_transcription.strip() and self.callback:
                        self.callback(current_transcription.strip())

                    # Clear the buffer
                    audio_buffer = []
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Transcription error: {e}")
                break
