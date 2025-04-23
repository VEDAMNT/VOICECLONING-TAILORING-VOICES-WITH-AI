# evaluate_models.py
# A basic script to demonstrate evaluation of Tacotron 2 and HiFi-GAN models using MOS, Error Rates, and Similarity (CPU version)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf
import logging
from typing import List, Tuple
from speechbrain.pretrained import SepformerSeparation
from resemblyzer import VoiceEncoder, preprocess_wav
from jiwer import wer, cer

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set device to CPU
device = "cpu"
logger.info("Running on CPU (CUDA not available).")

# Utility: Text to Sequence (from tacotron2.py)
def text_to_sequence(text: str, symbols: List[str]) -> List[int]:
    """Convert text to a sequence of integers."""
    sequence = []
    for char in text.lower():
        if char in symbols:
            sequence.append(symbols.index(char))
    sequence.append(symbols.index("<EOS>"))  # End of sequence token
    logger.info(f"Converted text to sequence: {sequence}")
    return sequence

# Utility: Mel-Spectrogram Extraction (from tacotron2.py and hifigan.py)
def get_mel_spectrogram(audio_path: str, sample_rate: int = 22050, n_mels: int = 80) -> np.ndarray:
    """Extract mel-spectrogram from an audio file."""
    y, sr = librosa.load(audio_path, sr=sample_rate)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=8000)
    mel = np.log(mel + 1e-9)  # Log mel-spectrogram
    logger.info(f"Extracted mel-spectrogram from {audio_path}, shape: {mel.shape}")
    return mel

# Simplified LocationSensitiveAttention
class LocationSensitiveAttention(nn.Module):
    def __init__(self, attention_dim: int, encoder_dim: int, decoder_dim: int):
        super(LocationSensitiveAttention, self).__init__()
        self.query_layer = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.memory_layer = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.v = nn.Parameter(torch.randn(attention_dim))
        self.score_mask_value = -float("inf")

    def forward(self, query: torch.Tensor, memory: torch.Tensor, attention_weights_cum: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention weights and context vector (simplified)."""
        processed_query = self.query_layer(query.unsqueeze(1))  # [batch_size, 1, attention_dim]
        processed_memory = self.memory_layer(memory)  # [batch_size, seq_len, attention_dim]
        scores = torch.tanh(processed_query + processed_memory)  # [batch_size, seq_len, attention_dim]
        scores = torch.sum(scores * self.v, dim=-1)  # [batch_size, seq_len]
        alignments = F.softmax(scores, dim=-1)  # [batch_size, seq_len]
        context = torch.bmm(alignments.unsqueeze(1), memory)  # [batch_size, 1, encoder_dim]
        return context.squeeze(1), alignments

# Tacotron 2 Model (simplified from tacotron2.py)
class Tacotron2(nn.Module):
    def __init__(self, n_symbols: int, embedding_dim: int = 512, encoder_dim: int = 512, decoder_dim: int = 1024, n_mels: int = 80):
        super(Tacotron2, self).__init__()
        self.n_mels = n_mels
        self.embedding = nn.Embedding(n_symbols, embedding_dim)
        self.embedding.weight.data.normal_(0, 0.3)
        self.encoder = nn.ModuleList([
            nn.Conv1d(embedding_dim, encoder_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(encoder_dim),
            nn.LSTM(encoder_dim, encoder_dim // 2, num_layers=3, bidirectional=True, batch_first=True)
        ])
        self.decoder_prenet = nn.Sequential(
            nn.Linear(n_mels, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.attention = LocationSensitiveAttention(128, encoder_dim, decoder_dim)
        self.decoder_rnn = nn.LSTMCell(256 + encoder_dim, decoder_dim)
        self.decoder_output = nn.Linear(decoder_dim + encoder_dim, n_mels * 3)
        self.postnet = nn.Sequential(
            nn.Conv1d(n_mels, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Conv1d(512, n_mels, kernel_size=5, padding=2)
        )
        logger.info("Tacotron2 model initialized.")

    def forward(self, text: torch.Tensor, mels: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.embedding(text).transpose(1, 2)
        encoder_conv = F.relu(self.encoder[0](embedded))
        encoder_conv = self.encoder[1](encoder_conv)
        encoder_output, _ = self.encoder[2](encoder_conv.transpose(1, 2))
        
        batch_size = text.size(0)
        seq_len = text.size(1)  # Length of the input sequence
        max_len = mels.size(2) if mels is not None else 300  # Length of the output mel-spectrogram

        decoder_input = torch.zeros(batch_size, self.n_mels, device=text.device)
        hidden, cell = torch.zeros(batch_size, 1024, device=text.device), torch.zeros(batch_size, 1024, device=text.device)
        attention_weights_cum = torch.zeros(batch_size, 1, seq_len, device=text.device)  # Match seq_len, not max_len
        mel_outputs, gate_outputs = [], []

        for t in range(max_len):
            prenet_out = self.decoder_prenet(decoder_input)
            context, alignments = self.attention(hidden, encoder_output, attention_weights_cum)
            attention_weights_cum += alignments.unsqueeze(1)  # Now shapes match: [batch_size, 1, seq_len]
            rnn_input = torch.cat([prenet_out, context], dim=-1)
            hidden, cell = self.decoder_rnn(rnn_input, (hidden, cell))
            decoder_out = torch.cat([hidden, context], dim=-1)
            mel_out = self.decoder_output(decoder_out).view(batch_size, 3, self.n_mels)
            gate_out = torch.sigmoid(mel_out[:, :, -1])
            mel_outputs.append(mel_out)
            gate_outputs.append(gate_out)
            decoder_input = mels[:, :, t] if mels is not None else mel_out[:, -1, :]

        mel_outputs = torch.cat(mel_outputs, dim=1).transpose(1, 2)
        gate_outputs = torch.cat(gate_outputs, dim=1)
        mel_postnet = self.postnet(mel_outputs) + mel_outputs
        return mel_outputs, mel_postnet, gate_outputs

    def inference(self, text: torch.Tensor, max_len: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            mel_outputs, mel_postnet, _ = self.forward(text, mels=None)
        return mel_outputs, mel_postnet

# HiFi-GAN Generator (simplified from hifigan.py)
class HiFiGANGenerator(nn.Module):
    def __init__(self, in_channels: int = 80, upsample_rates: List[int] = [8, 8, 2, 2]):
        super(HiFiGANGenerator, self).__init__()
        self.conv_pre = nn.Conv1d(in_channels, 512, kernel_size=7, padding=3)
        self.upsample = nn.ModuleList()
        for i, rate in enumerate(upsample_rates):
            self.upsample.append(
                nn.Sequential(
                    nn.ConvTranspose1d(512 // (2 ** i), 512 // (2 ** (i + 1)), kernel_size=rate * 2, stride=rate, padding=rate // 2),
                    nn.LeakyReLU(0.2),
                    nn.Conv1d(512 // (2 ** (i + 1)), 512 // (2 ** (i + 1)), kernel_size=3, padding=1)
                )
            )
        self.conv_post = nn.Conv1d(512 // (2 ** len(upsample_rates)), 1, kernel_size=7, padding=3)
        logger.info("HiFiGAN Generator initialized.")

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(mel)
        for upsample in self.upsample:
            x = upsample(x)
        x = self.conv_post(x)
        return torch.tanh(x)

# Evaluation Utilities
# 1. Error Rates (WER and CER) using SpeechBrain ASR
def compute_error_rates(generated_audio_path: str, ground_truth_text: str, sample_rate: int = 22050) -> Tuple[float, float]:
    """Compute WER and CER by transcribing the generated audio and comparing with ground truth text."""
    # Load SpeechBrain ASR model
    asr_model = SepformerSeparation.from_hparams(source="speechbrain/sepformer-wham", savedir="pretrained_models/sepformer-wham")
    # Transcribe the generated audio
    audio, sr = librosa.load(generated_audio_path, sr=sample_rate)
    sf.write("temp.wav", audio, sr)  # Save temporarily for SpeechBrain
    transcription = asr_model.separate_file("temp.wav").transcribe()
    # Compute WER and CER
    ground_truth = ground_truth_text.lower()
    hypothesis = transcription.lower()
    wer_score = wer(ground_truth, hypothesis)
    cer_score = cer(ground_truth, hypothesis)
    return wer_score, cer_score

# 2. Similarity using Resemblyzer
def compute_similarity(generated_audio_path: str, true_audio_path: str, sample_rate: int = 22050) -> float:
    """Compute cosine similarity between speaker embeddings of generated and true audio."""
    encoder = VoiceEncoder()
    # Load and preprocess audio
    gen_wav = preprocess_wav(generated_audio_path)
    true_wav = preprocess_wav(true_audio_path)
    # Extract embeddings
    gen_embedding = encoder.embed_utterance(gen_wav)
    true_embedding = encoder.embed_utterance(true_wav)
    # Compute cosine similarity
    similarity = np.dot(gen_embedding, true_embedding) / (np.linalg.norm(gen_embedding) * np.linalg.norm(true_embedding))
    return similarity

# 3. MOS (Simulated)
def simulate_mos(generated_audio_path: str) -> None:
    """Simulate MOS by prompting the user to listen and score the audio."""
    logger.info(f"Please listen to the generated audio at {generated_audio_path}.")
    logger.info("Rate the audio on a scale of 1 to 5 for the following criteria:")
    logger.info("- Clarity (1 = unintelligible, 5 = crystal clear)")
    logger.info("- Naturalness (1 = robotic, 5 = human-like)")
    logger.info("- Accent Match (1 = wrong accent, 5 = matches desired accent)")
    logger.info("Average the scores to get the MOS.")

# Generate Audio (Tacotron 2 + HiFi-GAN)
def generate_audio(tacotron2: Tacotron2, hifigan: HiFiGANGenerator, text: str, symbols: List[str], output_path: str = "generated_audio.wav", sample_rate: int = 22050) -> np.ndarray:
    """Generate audio using Tacotron 2 and HiFi-GAN."""
    # Step 1: Generate mel-spectrogram with Tacotron 2
    sequence = text_to_sequence(text, symbols)
    sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(device)
    tacotron2.eval()
    with torch.no_grad():
        _, mel_postnet, _ = tacotron2.inference(sequence_tensor)
        mel = mel_postnet.squeeze(0).cpu().numpy()

    # Step 2: Generate audio with HiFi-GAN
    hifigan.eval()
    with torch.no_grad():
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).to(device)
        audio = hifigan(mel_tensor)
        audio = audio.squeeze().cpu().numpy()
        sf.write(output_path, audio, sample_rate)
    logger.info(f"Generated audio saved to {output_path}")
    return mel

# Main Evaluation Function
def evaluate_models(tacotron2: Tacotron2, hifigan: HiFiGANGenerator, text: str, true_audio_path: str, symbols: List[str]) -> None:
    """Evaluate the models using MOS, Error Rates, and Similarity."""
    # Generate audio
    generated_audio_path = "generated_audio.wav"
    mel = generate_audio(tacotron2, hifigan, text, symbols, output_path=generated_audio_path)

    # 1. Compute Error Rates (WER and CER)
    logger.info("Computing Error Rates (WER and CER)...")
    wer_score, cer_score = compute_error_rates(generated_audio_path, text)
    logger.info(f"Word Error Rate (WER): {wer_score:.4f}")
    logger.info(f"Character Error Rate (CER): {cer_score:.4f}")

    # 2. Compute Similarity
    logger.info("Computing Similarity...")
    similarity_score = compute_similarity(generated_audio_path, true_audio_path)
    logger.info(f"Speaker Similarity (Cosine): {similarity_score:.4f}")

    # 3. Simulate MOS
    logger.info("Simulating MOS...")
    simulate_mos(generated_audio_path)

# Main Execution
if __name__ == "__main__":
    # Define symbols
    symbols = ["<PAD>", "<EOS>"] + list("abcdefghijklmnopqrstuvwxyz ")

    # Initialize models
    tacotron2 = Tacotron2(n_symbols=len(symbols)).to(device)
    hifigan = HiFiGANGenerator().to(device)

    # Load pre-trained weights (uncomment if you have saved models)
    # tacotron2.load_state_dict(torch.load("tacotron2_model.pth"))
    # hifigan.load_state_dict(torch.load("hifigan_generator.pth"))

    # Test data
    test_text = "Hello, this is a test."
    true_audio_path = "dummy_audio.wav"  # Replace with a real audio file

    # Evaluate models
    logger.info("Starting evaluation...")
    evaluate_models(tacotron2, hifigan, test_text, true_audio_path, symbols)
    logger.info("Evaluation completed.")