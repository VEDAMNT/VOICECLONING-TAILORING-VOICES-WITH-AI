import librosa
import numpy as np
import soundfile as sf
import os
import csv

def preprocess_audio(audio_path, output_dir="data"):
    # Load audio and resample to 22,050 Hz, mono
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output WAV path
    output_wav = os.path.join(output_dir, "user_audio.wav")
    sf.write(output_wav, y, 22050, subtype="PCM_16")
    
    # Your transcription
    text = "Hello this is Vedant Sandesh Dalvi and I am currently doing project based on voice cloning tailoring voices with artificial intelligence this is My voice First voice audio number 1"
    
    # Write to transcription.csv with ID
    csv_path = os.path.join(output_dir, "transcription.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ID", "wav_path", "text"])  # Header with ID
        writer.writerow(["sample1", output_wav, text])  # Data with unique ID
    
    print(f"Preprocessed audio saved to {output_wav}")
    print(f"Transcription saved to {csv_path}")

if __name__ == "__main__":
    preprocess_audio("C:/Users/dalvi/Desktop/VOICECLONING/data/user_voice.wav")