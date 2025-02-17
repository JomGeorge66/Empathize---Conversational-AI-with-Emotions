import torch
import torchaudio
import argparse
import sounddevice as sd
from zonos.model import Zonos, DEFAULT_BACKBONE_CLS as ZonosBackbone
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

def load_model(model_choice: str):
    """
    Loads the Zonos model if it isn't already loaded.
    """
    print(f"Loading {model_choice} model...")
    model = Zonos.from_pretrained(model_choice, device=device)
    model.requires_grad_(False).eval()
    print(f"{model_choice} model loaded successfully!")
    return model

def progress_callback(frame: torch.Tensor, step: int, total_steps: int) -> bool:
    """
    Simple callback to print progress.
    """
    print(f"Generation progress: {step}/{total_steps} steps", end="\r")
    return True

def main():
    parser = argparse.ArgumentParser(description="Zonos TTS with Emotion Control")
    parser.add_argument("--model_choice", type=str, default="Zyphra/Zonos-v0.1-transformer",
                        help="Model variant to use")
    parser.add_argument("--text", type=str, default="Hello, this is a test of the Zonos TTS with emotion control.",
                        help="Text to synthesize")
    parser.add_argument("--language", type=str, default="en-us",
                        help="Language code for synthesis (e.g., en-us)")
    
    # Emotion conditioning (8 dimensions: Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral)
    parser.add_argument("--happiness", type=float, default=1.0, help="Happiness value (0.0 to 1.0)")
    parser.add_argument("--sadness", type=float, default=0.05, help="Sadness value (0.0 to 1.0)")
    parser.add_argument("--disgust", type=float, default=0.05, help="Disgust value (0.0 to 1.0)")
    parser.add_argument("--fear", type=float, default=0.05, help="Fear value (0.0 to 1.0)")
    parser.add_argument("--surprise", type=float, default=0.05, help="Surprise value (0.0 to 1.0)")
    parser.add_argument("--anger", type=float, default=0.05, help="Anger value (0.0 to 1.0)")
    parser.add_argument("--other", type=float, default=0.1, help="Other emotion value (0.0 to 1.0)")
    parser.add_argument("--neutral", type=float, default=0.2, help="Neutral emotion value (0.0 to 1.0)")
    
    # Other conditioning parameters
    parser.add_argument("--vqscore", type=float, default=0.78,
                        help="VQScore for audio quality (0.5 to 0.8 recommended)")
    parser.add_argument("--fmax", type=float, default=24000.0,
                        help="Maximum frequency (Hz) for the audio (e.g., 22050 or 24000)")
    parser.add_argument("--pitch_std", type=float, default=45.0,
                        help="Standard deviation of pitch (wider for more expressive speech)")
    parser.add_argument("--speaking_rate", type=float, default=15.0,
                        help="Number of phonemes to read per second")
    parser.add_argument("--dnsmos", type=float, default=4.0,
                        help="DNSMOS overall score (1 to 5, use 4.0 for clean neutral English)")
    parser.add_argument("--cfg_scale", type=float, default=2.0,
                        help="CFG scale for generation (controls adherence to conditioning)")
    parser.add_argument("--min_p", type=float, default=0.15,
                        help="min_p parameter for sampling")
    parser.add_argument("--seed", type=int, default=420, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model_choice)
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Build the emotion tensor (8-dimensional)
    emotion_values = [args.happiness, args.sadness, args.disgust, args.fear,
                      args.surprise, args.anger, args.other, args.neutral]
    emotion_tensor = torch.tensor(emotion_values, device=device)
    
    # Build the vqscore tensor (repeat value for each of the 8 dimensions)
    vq_tensor = torch.tensor([args.vqscore] * 8, device=device).unsqueeze(0)
    
    # Create the conditioning dictionary.
    cond_dict = make_cond_dict(
        text=args.text,
        language=args.language,
        speaker=None,  # Using default voice, no speaker cloning
        emotion=emotion_tensor,
        vqscore_8=vq_tensor,
        fmax=args.fmax,
        pitch_std=args.pitch_std,
        speaking_rate=args.speaking_rate,
        dnsmos_ovrl=args.dnsmos,
        speaker_noised=False,
        device=device,
        unconditional_keys=[],
    )
    conditioning = model.prepare_conditioning(cond_dict)
    
    max_new_tokens = 86 * 30  # 30-second generation limit
    
    print("Generating audio...")
    codes = model.generate(
        prefix_conditioning=conditioning,
        audio_prefix_codes=None,
        max_new_tokens=max_new_tokens,
        cfg_scale=args.cfg_scale,
        batch_size=1,
        sampling_params=dict(min_p=args.min_p),
        callback=progress_callback,
    )
    
    # Decode the generated codes into waveform
    wav_out = model.autoencoder.decode(codes).cpu().detach()
    sr_out = model.autoencoder.sampling_rate
    if wav_out.dim() == 2 and wav_out.size(0) > 1:
        wav_out = wav_out[0:1, :]
    
    # Play audio
    print("Playing generated audio...")
    sd.play(wav_out.squeeze().numpy(), sr_out)
    sd.wait()
    
if __name__ == "__main__":
    main()
