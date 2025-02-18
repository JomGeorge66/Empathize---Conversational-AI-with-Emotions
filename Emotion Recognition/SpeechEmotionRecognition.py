from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import torch
import librosa
import numpy as np

class EmotionRecognizer:
    def __init__(self):
        # Load the saved model and processor
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained('Emotion Recognition/trained_model')
        self.processor = Wav2Vec2Processor.from_pretrained('Emotion Recognition/trained_model')

        # Set the model to evaluation mode
        self.model.eval()

        # Use CPU by default to avoid CUDA issues
        self.device = torch.device('cpu')
        self.model.to(self.device)

    def predict_emotion(self, audio_np):
        """
        Predict emotion from numpy array of audio data
        
        Args:
            audio_np: Numpy array of audio data (already at 16kHz sample rate)
        Returns:
            predicted_label: String containing the predicted emotion
        """
        try:
            # Preprocess the audio
            inputs = self.processor(
                audio_np, 
                sampling_rate=16000, 
                return_tensors='pt', 
                padding=True, 
                truncation=True,
                max_length=32000
            )

            # Get the input values
            input_values = inputs.input_values.squeeze().to(self.device)

            # Perform inference
            with torch.no_grad():
                outputs = self.model(input_values.unsqueeze(0))
                logits = outputs.logits

            # Get predicted class
            predicted_class = logits.argmax(dim=-1).item()

            # Map predicted class to label
            inverse_label_map = {
                0: 'disgust', 1: 'anger', 2: 'sad', 3: 'happy',
                4: 'surprise', 5: 'fear', 6: 'neutral'
            }
            return inverse_label_map[predicted_class]
        except Exception as e:
            print(f"Emotion prediction error: {e}")
            return 'neutral'  # Return neutral as fallback

    def predict_from_file(self, emotion_audio_path):
        """
        Predict emotion from an audio file path (kept for backwards compatibility)
        """
        try:
            # Load the audio file
            speech, _ = librosa.load(emotion_audio_path, sr=16000)
            return self.predict_emotion(speech)
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return 'neutral'

# For backwards compatibility
model = EmotionRecognizer()
def predict_audio(audio_path):
    return model.predict_from_file(audio_path)

# Example of using the model for prediction
emotion_audio_path = 'Emotion Recognition/Neural Network Training Files/Datasets/archive/tess toronto emotional speech set data/TESS Toronto emotional speech set data/OAF_happy/OAF_bar_happy.wav'
predicted_label = predict_audio(emotion_audio_path)
print(f"Predicted Emotion: {predicted_label}")
