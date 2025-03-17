import os
import torch
from TTS.api import TTS
import numpy as np
import scipy.io.wavfile  # Import without alias to avoid conflicts
import soundfile as sf
from pathlib import Path

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize XTTS v2 model
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device)

def preprocess_audio(input_path, output_path, sample_rate=22050):
    """Ensure audio is mono, 22kHz WAV."""
    import librosa
    try:
        y, sr = librosa.load(input_path, sr=sample_rate, mono=True)
        if len(y) == 0:
            raise ValueError("Audio file is empty")
        sf.write(output_path, y, sample_rate, subtype="PCM_16")
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise ValueError(f"Failed to write {output_path}")
        return output_path
    except Exception as e:
        print(f"Preprocessing error for {input_path}: {e}")
        raise

def generate_audio(text, speaker_wavs, output_path="output.wav", language="en"):
    """Clone voice from a list of speaker_wavs and synthesize text."""
    if not text or not text.strip():
        text = "Hello, this is a default message."
    if not isinstance(speaker_wavs, list):
        speaker_wavs = [speaker_wavs]
    
    print(f"Text: {text}, Speaker WAVs: {speaker_wavs}")
    processed_wavs = []
    for i, wav in enumerate(speaker_wavs):
        processed_path = f"{output_path.replace('.wav', '')}_processed_{i}.wav"
        processed_wavs.append(preprocess_audio(wav, processed_path))
    
    try:
        wav_data = tts.tts(text=text, speaker_wav=processed_wavs, language=language)
        audio_int16 = (np.array(wav_data) * 32768).astype("int16")
        scipy.io.wavfile.write(output_path, 22050, audio_int16)  # Use full module path
        print(f"Generated audio at {output_path}")
        return output_path
    except Exception as e:
        print(f"TTS error: {e}")
        raise

if __name__ == "__main__":
    speaker_wavs = ["user_audio.wav", "user_voice.wav"]  # Replace with your audio files
    text = "Hello, this is my cloned voice!"
    print("Generating audio...")
    generate_audio(text, speaker_wavs, "test_output.wav")
    print("Done!")