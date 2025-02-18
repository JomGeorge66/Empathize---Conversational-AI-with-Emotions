# Empathize - Conversation AI with Emotions


Empathize is a modular conversational AI platform that seamlessly blends real-time speech emotion recognition with expressive text-to-speech (TTS) synthesis. By capturing user speech via microphone, analyzing its emotional content, and generating empathetic responses, Empathize creates a truly human-like conversational experience.

## Important Note
- **Speech to text model and TTS model could not be loaded together due to VRAM limitations of my system.** 6GB of VRAM was not sufficient. **CUDA was running out of memory.**
- **However STT and TTS with emotions are working individually, and they can be integrated and implemented on a powerful system.**
- Even while testing, TTS with emotions, CUDA was running out of memory for longer conversations.


**Note: The files are being uploaded and are not completed yet.**

## Watch the demo here:
https://www.loom.com/share/4944e6318d764b7994828571b40b4fee?sid=6291b024-b4f4-499f-971c-76d8063e587d
### Also watch a video about the project overview:
https://youtu.be/6_rKNqf_Gf8

---
[![](https://mermaid.ink/img/pako:eNqNk29r2zAQxr_KoTFIIR7d0qRLXgycOP9K25XFMOi8F4p0jsUUyUjyslD63XdxXCfsxYhfGJ30e56709kvTFiJbMRybXei4C5AmmQG6Il_ZOxe_UZYlYiigKUpqwCdByWcLQtrEOJKKgsTXobK4VXGfkIUfYEx6ZKK6-iJhwKenBXovTIbOj8aj2tsQtjROQo2SvEPec-4D-ii74XyJbqrfxQJKaZbG5Q18A2F3RhVrzvJ42MXUseVQQm0kU5XK0h44B5DF4bDD4P3EAtROS72J9NJbTol0zo56Y0XTpUHzxZKamh2lvmer1G359P6fE7n8QZNUALiJXQm1gTyjOIdd0i1-tIajzBHg44fXE5VzI4Gx2BeBwtya1Dqp1XvFF1nUwXX0ORojRa1dnmqNDoQzmrN1xohTVfQebbGenigietTBctaeFcLSxoZHrpoZv61CjT0Fj2-fdiTYQy50nr0Ls9xLUTXU6pfSOH1uvfp5hwcN6C8wY95vwV7_HYt--fg5FIwuRScNiD28ms5bMGBGMqeOAdnl4Lzt64_y1vJW1Dc9Af94Tm4uNRxeWkzd_-_cNZlW3RbriT9zS8HYcZomFvM2IiWEnNeaRplZl4J5VWwq70RbBRchV3mbLUp2Cjn2lNUlZK-vUTxjePbdrfk5tnat_j1LzDuVok?type=png)](https://mermaid.live/edit#pako:eNqNk29r2zAQxr_KoTFIIR7d0qRLXgycOP9K25XFMOi8F4p0jsUUyUjyslD63XdxXCfsxYhfGJ30e56709kvTFiJbMRybXei4C5AmmQG6Il_ZOxe_UZYlYiigKUpqwCdByWcLQtrEOJKKgsTXobK4VXGfkIUfYEx6ZKK6-iJhwKenBXovTIbOj8aj2tsQtjROQo2SvEPec-4D-ii74XyJbqrfxQJKaZbG5Q18A2F3RhVrzvJ42MXUseVQQm0kU5XK0h44B5DF4bDD4P3EAtROS72J9NJbTol0zo56Y0XTpUHzxZKamh2lvmer1G359P6fE7n8QZNUALiJXQm1gTyjOIdd0i1-tIajzBHg44fXE5VzI4Gx2BeBwtya1Dqp1XvFF1nUwXX0ORojRa1dnmqNDoQzmrN1xohTVfQebbGenigietTBctaeFcLSxoZHrpoZv61CjT0Fj2-fdiTYQy50nr0Ls9xLUTXU6pfSOH1uvfp5hwcN6C8wY95vwV7_HYt--fg5FIwuRScNiD28ms5bMGBGMqeOAdnl4Lzt64_y1vJW1Dc9Af94Tm4uNRxeWkzd_-_cNZlW3RbriT9zS8HYcZomFvM2IiWEnNeaRplZl4J5VWwq70RbBRchV3mbLUp2Cjn2lNUlZK-vUTxjePbdrfk5tnat_j1LzDuVok)
## Overview

Empathize integrates several state-of-the-art components:

- **Live Speech Input:** Captures audio from a microphone.
- **Dual-Path Processing:**
  - **Speech-to-Text:** Uses Faster-Whisper to transcribe live speech.
  - **Emotion Recognition:** Leverages a deep neural network (DNN) trained on the Toronto Emotional Speech Set (via Kaggle) with Wav2Vec, achieving 99.6% accuracy.
- **Agentic AI:** Consumes the transcription and detected emotion to generate an empathetic, context-aware response.
- **Emotion-Controllable TTS:** Employs the open-source Zonos model to synthesize speech that reflects the desired emotional tone.

Supported emotions include: **happy, sad, anger, surprise, disgust, fear,** and **neutral**.

---

## System Architecture

1. **Input & Preprocessing:**
   - **Microphone Capture:** Live audio is captured.
   - **Parallel Analysis:**
     - **Faster-Whisper Module:** Transcribes the speech.
     - **Emotion DNN:** Predicts emotion from the speech signal.
2. **Agentic AI Response Generation:**
   - Combines the transcribed text with the detected emotion.
   - Generates a response that matches the conversational context and desired emotional output.
3. **Emotion-Controllable TTS:**
   - Uses an 8-dimensional emotion conditioning vector to control vocal expression.
   - Fine-tunes audio parameters such as speaking rate, pitch variability, and overall quality.

---

## Key Components

### Speech Emotion Recognition

- **Model Training:** The emotion DNN was trained on the Toronto Emotional Speech Set from Kaggle and integrated with Wav2Vec for high accuracy.
- **Assets:** All training scripts and pretrained models are available in the `assets/` directory.

### Speech-to-Text Transcription

- **Whisper Integration:** Faster Whisper runs in parallel with the emotion recognition module, ensuring real-time transcription.

### Agentic AI for Empathetic Conversations

- **Response Generation:** Based on the transcribed text and emotion data, the Agentic AI crafts responses that are not only contextually relevant but also emotionally resonant.
- **Emotion Parameter Guidance:** These responses include specific emotion parameters used to steer the TTS output.

## Advanced Text-to-Speech Synthesis

Empathize employs Zonos TTS—a flexible, open-source text-to-speech engine—to transform generated text into natural, expressive speech. This system is designed to provide detailed control over the vocal output through an array of configuration parameters:

- **Core Inputs:**
  - **Text Input (`--text`):** The raw text for conversion.
  - **Emotion Conditioning Vector:** An 8-dimensional vector specifying the emotional tone. The order is:
    1. Happiness (`--happiness`)
    2. Sadness (`--sadness`)
    3. Disgust (`--disgust`)
    4. Fear (`--fear`)
    5. Surprise (`--surprise`)
    6. Anger (`--anger`)
    7. Other (`--other`)
    8. Neutral (`--neutral`)
  
  Adjust these values (ranging from 0.0 to 1.0) to modulate the emotional expressiveness. For example, setting a higher value for `--happiness` and lower for `--sadness` and `--anger` yields a more cheerful tone.

- **Audio Quality & Expressiveness Controls:**
  - **VQScore (`--vqscore`):** Controls overall audio quality (recommended value around 0.78).
  - **Maximum Frequency (`--fmax`):** Sets the maximum frequency of the audio (recommended: 22050 or 24000 Hz).
  - **Pitch Standard Deviation (`--pitch_std`):** Adjusts the variability in pitch. Use lower values (20–45) for natural speech or higher (60–150) for more expressive output.
  - **Speaking Rate (`--speaking_rate`):** Determines the speed (number of phonemes per second) at which speech is generated.
  - **DNSMOS Overall (`--dnsmos`):** A quality estimation metric (scale 1 to 5; 4.0 is recommended for clean, neutral speech).
  - **CFG Scale (`--cfg_scale`):** Controls how strictly the model adheres to the provided conditioning.
  - **Sampling Parameter (`--min_p`):** Influences the randomness versus determinism in the sampling process.
  - **Random Seed (`--seed`):** For reproducible outputs.

This unified configuration allows you to precisely tailor the TTS output, ensuring that the synthesized speech matches both the content and the intended emotional tone of the conversation.

---

## Integration Options

Your AI agent can interact with Empathize in two common ways:

### 1. Subprocess Invocation

Invoke the TTS script from your agent using a subprocess call. For example:

```python
import subprocess

subprocess.run([
    "python", "tts_emotion.py",
    "--text", "Your AI-generated sentence",
    "--happiness", "0.9",
    "--sadness", "0.1",
    # ... include other emotion and audio parameters as needed ...
])
```

## Installation

#### System requirements

- **Operating System:** Linux (preferably Ubuntu 22.04/24.04), macOS
- **GPU:** 6GB+ VRAM, Hybrid additionally requires a 3000-series or newer Nvidia GPU

Note: Zonos can also run on CPU provided there is enough free RAM. However, this will be a lot slower than running on a dedicated GPU, and likely won't be sufficient for interactive use.

#### System dependencies

Zonos depends on the eSpeak library phonemization. You can install it on Ubuntu with the following command:

```bash
apt install -y espeak-ng # For Ubuntu
# brew install espeak-ng # For MacOS
```

#### Python dependencies

##### Clone this repository
```bash
git clone https://github.com/JomGeorge66/Empathize---Conversational-AI-with-Emotions.git
cd Empathize---Conversational-AI-with-Emotions/Zonos/
```
##### Installing into a new uv virtual environment (recommended)

```bash
uv sync
uv pip install -e .
```
#### Activate the Virtual enviornment in Zonos folder in your Code editor (Important)
Make sure you give your enviornment path in the 'Select Interpreter' option and enter Interpreter path.
Path:
```bash
Zonos/.venv/bin/python3.12
```
##### Installing all the dependencies
```bash
cd .. #Go back from the Zonos folder
pip install -r requirements.txt
```
### Running The Program
####Navigate to Zonos & Run empathizeai.py

```bash
cd Zonos
python3 empathiseai.py
```
