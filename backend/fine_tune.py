import os
import torch
import torchaudio.transforms as T
from speechbrain.core import Brain
from speechbrain.inference.TTS import Tacotron2
from speechbrain.dataio.dataset import DynamicItemDataset
import soundfile as sf
import numpy as np

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load pre-trained Tacotron2
print("Loading pre-trained Tacotron2...")
tacotron2 = Tacotron2.from_hparams(
    source="speechbrain/tts-tacotron2-ljspeech",
    savedir="pretrained_models/tacotron2",
    run_opts={"device": "cpu"}
)

# Define a simple Brain class for fine-tuning
class TTSBrain(Brain):
    def compute_forward(self, batch, stage):
        wavs = batch["wav"].to(self.device)
        text_seq = batch["text_seq"].to(self.device)
        mel_target = batch["mel_target"].to(self.device)
        mel_output, mel_length, alignment = self.modules.model(text_seq, mel_target)
        return mel_output, mel_length, alignment

    def compute_objectives(self, predictions, batch, stage):
        mel_output, mel_length, _ = predictions
        mel_target = batch["mel_target"].to(self.device)
        loss = torch.nn.functional.mse_loss(mel_output, mel_target)
        return loss

    def on_fit_start(self):
        self.modules.model.train()

# Prepare data
audio_path = "data/user_voice.wav"
text = open("data/transcript.txt", "r").read().strip()
print(f"Text: '{text}'")

# Read audio
wav, sr = sf.read(audio_path)
if sr != 22050:
    raise ValueError("Audio must be 22,050 Hz")
wav = torch.tensor(wav).float()  # Shape: (T,)

# Compute mel-spectrogram
mel_spec = T.MelSpectrogram(
    sample_rate=22050,
    n_fft=tacotron2.hparams.n_fft,
    win_length=tacotron2.hparams.win_length,
    hop_length=tacotron2.hparams.hop_length,
    n_mels=80
)
mel_target = mel_spec(wav.unsqueeze(0)).transpose(1, 2)  # Shape: (1, n_mels, frames)
print(f"Mel-spectrogram shape: {mel_target.shape}")

# Tokenize text
text_seq = torch.tensor(tacotron2.tokenizer.encode_text(text)).unsqueeze(0)  # Shape: (1, seq_len)
print(f"Text sequence shape: {text_seq.shape}")

# Create a simple dataset
data = [{"wav": wav, "text_seq": text_seq[0], "mel_target": mel_target[0], "id": "user_sample"}]
dataset = DynamicItemDataset.from_list(data)

# Initialize Brain
tts_brain = TTSBrain(
    modules={"model": tacotron2.mods["model"]},
    opt_class=lambda x: torch.optim.Adam(x, lr=1e-4),
    hparams={"epochs": 50},
    run_opts={"device": "cpu"}
)

# Fine-tune
print("Starting fine-tuning...")
tts_brain.fit(
    epoch_counter=range(1, 51),
    train_set=dataset,
    train_loader_kwargs={"batch_size": 1, "shuffle": False}
)

# Save fine-tuned model
torch.save(tacotron2.mods["model"].state_dict(), "fine_tuned_tacotron2.pth")
print("Fine-tuning complete! Saved to fine_tuned_tacotron2.pth")