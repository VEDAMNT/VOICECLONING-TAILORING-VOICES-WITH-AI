import os
import torch
from TTS.api import TTS
import numpy as np
import scipy.io.wavfile
import soundfile as sf
from pathlib import Path
from pydub import AudioSegment
import librosa

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device)

print("Supported languages:", tts.languages)

SAMPLED_FOLDER = "sampled"
Path(SAMPLED_FOLDER).mkdir(exist_ok=True)

def preprocess_audio(input_path, output_path, sample_rate=22050):
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
        yield output_path
    except Exception as e:
        print(f"Preprocessing error for {input_path}: {e}")
        raise

def apply_pre_emphasis(signal, coef=0.95):
    """
    Apply pre-emphasis filter to the signal to boost higher frequencies.
    signal: numpy array of audio samples
    coef: pre-emphasis coefficient (typically between 0.9 and 1.0)
    """
    # Ensure signal is a numpy array
    signal = np.asarray(signal)
    # Apply pre-emphasis: y[n] = x[n] - coef * x[n-1]
    emphasized_signal = np.append(signal[0], signal[1:] - coef * signal[:-1])
    return emphasized_signal

def generate_audio(text, speaker_wavs, output_path="output.wav", language="en", speed=1.0, pitch=0.0, voiceProfile="default"):
    if not text or not text.strip():
        text = "Hello, this is my cloned voice."
    if not isinstance(speaker_wavs, list):
        speaker_wavs = [speaker_wavs]
    if language not in ["en", "hi"]:
        raise ValueError("Unsupported language. Only 'en' (English) and 'hi' (Hindi) are supported.")
    
    print(f"Text: {text}, Speaker WAVs: {speaker_wavs}, Language: {language}, Speed: {speed}, Pitch: {pitch}")

    processed_wavs = []
    yield 0
    for i, wav in enumerate(speaker_wavs):
        processed_filename = f"{Path(output_path).stem}_processed_{i}.wav"
        processed_path = os.path.join(SAMPLED_FOLDER, processed_filename)
        progress_gen = preprocess_audio(wav, processed_path)
        for item in progress_gen:
            if isinstance(item, str):
                processed_wavs.append(item)
                yield item
            else:
                yield item
    
    yield 40
    try:
        yield 50
        wav_data = tts.tts(text=text, speaker_wav=processed_wavs, language=language)
        yield 60
        audio_int16 = (np.array(wav_data) * 32768).astype("int16")
        scipy.io.wavfile.write(output_path, 22050, audio_int16)
        
        # Apply speed adjustment using librosa (preserves pitch)
        if speed != 1.0:
            y, sr = librosa.load(output_path, sr=22050)
            y_stretched = librosa.effects.time_stretch(y, rate=speed)
            sf.write(output_path, y_stretched, sr, subtype="PCM_16")
        yield 70

        # Apply pitch adjustment
        if pitch != 0.0:
            y, sr = librosa.load(output_path, sr=22050)
            y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch)
            # Apply pre-emphasis to boost higher frequencies and improve clarity
            y_boosted = apply_pre_emphasis(y_shifted, coef=0.95)
            sf.write(output_path, y_boosted, sr, subtype="PCM_16")
        yield 80

        # Apply additional effects based on voiceProfile (e.g., volume, reverb) if implemented
        if voiceProfile == "deepFemale":
            # Example: Add subtle reverb for richness (if reverb is implemented)
            pass
        
        yield 90
        yield 100
        print(f"Generated audio at {output_path}")
        return output_path
    except Exception as e:
        print(f"TTS error: {e}")
        raise

if __name__ == "__main__":
    speaker_wavs = ["user_audio.wav"]
    text = "नमस्ते, यह एक परीक्षण है!"
    print("Generating audio in Hindi...")
    for progress in generate_audio(text, speaker_wavs, "test_output_hi.wav", language="hi", speed=0.95, pitch=-3.0):
        if isinstance(progress, int):
            print(f"Progress: {progress}%")
    print("Done!")