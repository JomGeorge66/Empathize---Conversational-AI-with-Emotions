# tts_emotions.py (modified)
import torch
import torchaudio
import sounddevice as sd
from zonos.model import Zonos, DEFAULT_BACKBONE_CLS as ZonosBackbone
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

class EmotionTTS:
    def __init__(self, model_choice="Zyphra/Zonos-v0.1-transformer"):
        self.device = device
        self.model = self._load_model(model_choice)
        self.default_params = {
            'vqscore': 0.78,
            'fmax': 24000.0,
            'pitch_std': 45.0,
            'speaking_rate': 15.0,
            'dnsmos': 4.0,
            'cfg_scale': 2.0,
            'min_p': 0.15,
            'seed': 420,
            'language': 'en-us'
        }

    def _load_model(self, model_choice):
        print(f"Loading {model_choice} model...")
        model = Zonos.from_pretrained(model_choice, device=self.device)
        model.requires_grad_(False).eval()
        print(f"{model_choice} model loaded successfully!")
        return model

    def _progress_callback(self, frame: torch.Tensor, step: int, total_steps: int) -> bool:
        print(f"Generation progress: {step}/{total_steps} steps", end="\r")
        return True

    def generate_speech(self, text: str, emotions: dict, audio_params: dict):
        # Merge parameters with defaults
        params = {**self.default_params, **audio_params}
        
        # Set random seed
        torch.manual_seed(params['seed'])
        
        # Build emotion tensor
        emotion_values = [
            emotions['happiness'],
            emotions['sadness'],
            emotions['disgust'],
            emotions['fear'],
            emotions['surprise'],
            emotions['anger'],
            emotions['other'],
            emotions['neutral']
        ]
        emotion_tensor = torch.tensor(emotion_values, device=self.device)
        
        # Build vqscore tensor
        vq_tensor = torch.tensor([params['vqscore']] * 8, device=self.device).unsqueeze(0)
        
        # Create conditioning dictionary
        cond_dict = make_cond_dict(
            text=text,
            language="en-us",
            speaker=None,
            emotion=emotion_tensor,
            vqscore_8=vq_tensor,
            fmax=params['fmax'],
            pitch_std=params['pitch_std'],
            speaking_rate=params['speaking_rate'],
            dnsmos_ovrl=params['dnsmos'],
            speaker_noised=False,
            device=self.device,
            unconditional_keys=[],
        )
        conditioning = self.model.prepare_conditioning(cond_dict)
        
        # Generate audio
        max_new_tokens = 86 * 30  # 30-second limit
        codes = self.model.generate(
            prefix_conditioning=conditioning,
            audio_prefix_codes=None,
            max_new_tokens=max_new_tokens,
            cfg_scale=params['cfg_scale'],
            batch_size=1,
            sampling_params=dict(min_p=params['min_p']),
            callback=self._progress_callback,
        )
        
        # Decode and return audio
        wav_out = self.model.autoencoder.decode(codes).cpu().detach()
        if wav_out.dim() == 2 and wav_out.size(0) > 1:
            wav_out = wav_out[0:1, :]
        
        return wav_out.squeeze().numpy(), self.model.autoencoder.sampling_rate

def main():
    # Example usage
    tts = EmotionTTS()
    
    # Example parameters
    emotions = {
        'happiness': 0.05,
        'sadness': 0.9,
        'disgust': 0.02,
        'fear': 0.1,
        'surprise': 0.05,
        'anger': 0.05,
        'other': 0.1,
        'neutral': 0.05
    }
    
    audio_params = {
        'pitch_std': 35.0,
        'speaking_rate': 12.0
    }
    
    audio, sr = tts.generate_speech(
        text="I'm so sorry for your loss...",
        emotions=emotions,
        audio_params=audio_params
    )
    
    print("Playing audio...")
    sd.play(audio, sr)
    sd.wait()

if __name__ == "__main__":
    main()
