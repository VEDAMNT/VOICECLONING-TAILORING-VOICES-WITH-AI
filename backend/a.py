import os
import torch
from TTS.api import TTS
import numpy as np
import scipy.io.wavfile
import soundfile as sf
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device)

# Print supported languages
print("Supported languages:", tts.languages)

def preprocess_audio(input_path, output_path, sample_rate=22050):
    import librosa
    try:
        yield 10
        y, sr = librosa.load(input_path, sr=sample_rate, mono=True)
        if len(y) == 0:
            raise ValueError("Audio file is empty")
        yield 20
        sf.write(output_path, y, sample_rate, subtype="PCM_16")
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise ValueError(f"Failed to write {output_path}")
        yield 30
        return output_path
    except Exception as e:
        print(f"Preprocessing error for {input_path}: {e}")
        raise

def generate_audio(text, speaker_wavs, output_path="output.wav", language="en"):
    if not text or not text.strip():
        text = "Hello, this is my cloned voice."
    if not isinstance(speaker_wavs, list):
        speaker_wavs = [speaker_wavs]
    
    print(f"Text: {text}, Speaker WAVs: {speaker_wavs}, Language: {language}")
    processed_wavs = []
    yield 0
    for i, wav in enumerate(speaker_wavs):
        processed_path = f"{output_path.replace('.wav', '')}_processed_{i}.wav"
        for progress in preprocess_audio(wav, processed_path):
            yield progress
        processed_wavs.append(processed_path)
    
    yield 40
    try:
        yield 50
        wav_data = tts.tts(text=text, speaker_wav=processed_wavs, language=language)
        yield 80
        audio_int16 = (np.array(wav_data) * 32768).astype("int16")
        yield 90
        scipy.io.wavfile.write(output_path, 22050, audio_int16)
        yield 100
        print(f"Generated audio at {output_path}")
        return output_path
    except Exception as e:
        print(f"TTS error: {e}")
        raise

if __name__ == "__main__":
    speaker_wavs = ["user_sample1.wav", "user_sample2.wav"]
    text = "Hello, this is a test!"
    print("Generating audio...")
    for progress in generate_audio(text, speaker_wavs, "test_output.wav"):
        print(f"Progress: {progress}%")
    print("Done!")