import torch
import torchaudio
from speechbrain.inference.TTS import Tacotron2
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.lobes.features import MFCC
import os

# Load pre-trained Tacotron 2
tacotron2 = Tacotron2.from_hparams(
    source="speechbrain/tts-tacotron2-ljspeech",
    savedir="pretrained_models/tacotron2"
)

# Prepare dataset
data_dir = "data"
dataset = DynamicItemDataset.from_csv(
    csv_path=os.path.join(data_dir, "transcription.csv")
)

# Define data pipeline with correct decorator syntax
def audio_pipeline(wav_path):
    signal, sr = torchaudio.load(wav_path)  # Load WAV
    mfcc = MFCC()(signal)  # Compute mel-spectrogram-like features
    return {"signal": signal, "mel": mfcc}
dataset.add_dynamic_item(func=audio_pipeline, takes=["wav_path"], provides=["signal", "mel"])

def text_pipeline(text):
    return {"text": text}
dataset.add_dynamic_item(func=text_pipeline, takes=["text"], provides=["text"])

dataset.set_output_keys(["id", "signal", "mel", "text"])

# Fine-tune
optimizer = torch.optim.Adam(tacotron2.parameters(), lr=1e-4)
for epoch in range(10):  # Small epochs for testing with one sample
    for batch in dataset:
        mel_output, mel_length, alignment = tacotron2.encode_text(batch["text"])
        # Simplified loss (mean squared error)
        loss = torch.mean((mel_output - batch["mel"]) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1} completed, Loss: {loss.item()}")

# Save fine-tuned model
tacotron2.hparams.save_path = "pretrained_models/tacotron2_finetuned"
tacotron2.save()