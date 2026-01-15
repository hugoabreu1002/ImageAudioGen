"""
Audio Regeneration with Synthetic Music using Autoencoder
Complete implementation to reconstruct audio stems

Requirements:
- Use synthetic music generated with harmonic compositions
- Audio preprocessing (Mel-Spectrogram)
- Implement model capable of regenerating audio
- Train and present examples
- Compare qualitatively and quantitatively
- Save and load model weights
- CLI interface for inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import os
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Optional, List
import argparse
import warnings

warnings.filterwarnings("ignore")


class AudioPreprocessor:
    """Audio preprocessor with Mel-Spectrogram"""

    def __init__(self, sample_rate: int = 16000, n_mels: int = 128, n_fft: int = 2048):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = n_fft // 4

        # Mel-Spectrogram transformation
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=self.hop_length,
            normalized=True,
        )

        # Inverse transformation
        self.griffin_lim = T.GriffinLim(
            n_fft=n_fft, hop_length=self.hop_length, n_iter=32
        )

        # Inverse Mel scale
        self.inverse_mel_scale = T.InverseMelScale(
            n_stft=n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=sample_rate,
        )

    def to_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert audio to Mel-Spectrogram

        Args:
            waveform: audio tensor [channels, samples]
        Returns:
            mel_spec: [channels, n_mels, time_steps]
        """
        # If mono, add channel dimension
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Convert to Mel-Spectrogram
        mel_spec = self.mel_transform(waveform)

        # Log scale
        mel_spec = torch.log(mel_spec + 1e-9)

        return mel_spec

    def from_mel_spectrogram(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Convert Mel-Spectrogram back to audio

        Args:
            mel_spec: [channels, n_mels, time_steps]
        Returns:
            waveform: [channels, samples]
        """
        # Remove log scale
        mel_spec = torch.exp(mel_spec) - 1e-9

        # Convert to linear spectrogram
        linear_spec = self.inverse_mel_scale(mel_spec)

        # Griffin-Lim for reconstruction
        waveform = self.griffin_lim(linear_spec)

        return waveform

    def normalize(self, mel_spec: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """Normalize Mel-Spectrogram"""
        mean = mel_spec.mean()
        std = mel_spec.std()
        normalized = (mel_spec - mean) / (std + 1e-8)
        return normalized, mean.item(), std.item()

    def denormalize(
        self, mel_spec: torch.Tensor, mean: float, std: float
    ) -> torch.Tensor:
        """Denormalize Mel-Spectrogram"""
        return mel_spec * std + mean


def generate_synthetic_music(
    sample_rate: int = 16000,
    duration: float = 10.0,
    composition_type: int = 1,
    n_harmonics: int = 5,
    base_freq: float = 220.0,
) -> torch.Tensor:
    """
    Generate synthetic music using harmonic compositions

    Args:
        sample_rate: Audio sample rate
        duration: Duration in seconds
        composition_type: Type of composition (1-3)
        n_harmonics: Number of harmonics to include
        base_freq: Base frequency for the composition

    Returns:
        Generated audio waveform as torch tensor
    """
    t = torch.linspace(0, duration, int(sample_rate * duration))

    if composition_type == 1:
        # Composition 1: Arpeggio-like progression with harmonics
        # Create a sequence of notes: C4, E4, G4, C5
        notes = [261.63, 329.63, 392.00, 523.25]  # C4, E4, G4, C5
        note_duration = duration / len(notes)

        waveform = torch.zeros_like(t)

        for i, freq in enumerate(notes):
            start_idx = int(i * note_duration * sample_rate)
            end_idx = int((i + 1) * note_duration * sample_rate)

            t_segment = t[start_idx:end_idx]
            note_wave = torch.zeros_like(t_segment)

            # Add fundamental and harmonics
            for h in range(1, n_harmonics + 1):
                amplitude = 1.0 / h  # Decreasing amplitude for higher harmonics
                harmonic_wave = amplitude * torch.sin(
                    2 * torch.pi * freq * h * t_segment
                )
                note_wave += harmonic_wave

            # Apply envelope (ADSR-like)
            attack_time = 0.1
            decay_time = 0.2
            sustain_level = 0.7
            release_time = 0.3

            segment_length = len(t_segment)
            attack_samples = int(attack_time * sample_rate)
            decay_samples = int(decay_time * sample_rate)
            release_samples = int(release_time * sample_rate)

            envelope = torch.ones(segment_length)

            # Attack
            if attack_samples > 0:
                envelope[:attack_samples] = torch.linspace(0, 1, attack_samples)

            # Decay and sustain
            if decay_samples > 0 and attack_samples + decay_samples < segment_length:
                decay_start = attack_samples
                decay_end = min(
                    attack_samples + decay_samples, segment_length - release_samples
                )
                if decay_end > decay_start:
                    envelope[decay_start:decay_end] = torch.linspace(
                        1, sustain_level, decay_end - decay_start
                    )

            # Release
            if release_samples > 0:
                release_start = max(0, segment_length - release_samples)
                envelope[release_start:] = torch.linspace(
                    sustain_level, 0, segment_length - release_start
                )

            waveform[start_idx:end_idx] = note_wave * envelope

    elif composition_type == 2:
        # Composition 2: Chord progression with complex harmonics
        # Create chords: C major, F major, G major, C major
        chords = [
            [261.63, 329.63, 392.00],  # C major
            [349.23, 440.00, 523.25],  # F major
            [392.00, 493.88, 587.33],  # G major
            [261.63, 329.63, 392.00],  # C major
        ]
        chord_duration = duration / len(chords)

        waveform = torch.zeros_like(t)

        for i, chord in enumerate(chords):
            start_idx = int(i * chord_duration * sample_rate)
            end_idx = int((i + 1) * chord_duration * sample_rate)

            t_segment = t[start_idx:end_idx]
            chord_wave = torch.zeros_like(t_segment)

            # Add each note in the chord with harmonics
            for freq in chord:
                note_wave = torch.zeros_like(t_segment)
                for h in range(1, n_harmonics + 1):
                    amplitude = 0.3 / h  # Softer amplitude
                    harmonic_wave = amplitude * torch.sin(
                        2 * torch.pi * freq * h * t_segment
                    )
                    note_wave += harmonic_wave
                chord_wave += note_wave

            # Apply envelope
            attack_time = 0.05
            decay_time = 0.1
            sustain_level = 0.8
            release_time = 0.2

            segment_length = len(t_segment)
            attack_samples = int(attack_time * sample_rate)
            decay_samples = int(decay_time * sample_rate)
            release_samples = int(release_time * sample_rate)

            envelope = torch.ones(segment_length)

            # Attack
            if attack_samples > 0:
                envelope[:attack_samples] = torch.linspace(0, 1, attack_samples)

            # Decay and sustain
            if decay_samples > 0 and attack_samples + decay_samples < segment_length:
                decay_start = attack_samples
                decay_end = min(
                    attack_samples + decay_samples, segment_length - release_samples
                )
                if decay_end > decay_start:
                    envelope[decay_start:decay_end] = torch.linspace(
                        1, sustain_level, decay_end - decay_start
                    )

            # Release
            if release_samples > 0:
                release_start = max(0, segment_length - release_samples)
                envelope[release_start:] = torch.linspace(
                    sustain_level, 0, segment_length - release_start
                )

            waveform[start_idx:end_idx] = chord_wave * envelope

    elif composition_type == 3:
        # Composition 3: Melody with counterpoint and rich harmonics
        # Create a more complex melody with multiple voices
        melody_notes = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
        counterpoint_notes = [
            130.81,
            146.83,
            164.81,
            174.61,
            196.00,
            220.00,
            246.94,
            261.63,
        ]

        note_duration = duration / len(melody_notes)

        waveform = torch.zeros_like(t)

        # Add melody
        for i, freq in enumerate(melody_notes):
            start_idx = int(i * note_duration * sample_rate)
            end_idx = int((i + 1) * note_duration * sample_rate)

            t_segment = t[start_idx:end_idx]
            note_wave = torch.zeros_like(t_segment)

            for h in range(1, n_harmonics + 1):
                amplitude = 0.4 / h
                harmonic_wave = amplitude * torch.sin(
                    2 * torch.pi * freq * h * t_segment
                )
                note_wave += harmonic_wave

            # Simple envelope for melody
            envelope = torch.exp(-torch.linspace(0, 3, len(t_segment)))
            waveform[start_idx:end_idx] += note_wave * envelope

        # Add counterpoint (softer)
        for i, freq in enumerate(counterpoint_notes):
            start_idx = int(
                (i + 0.5) * note_duration * sample_rate
            )  # Offset by half note
            end_idx = int((i + 1.5) * note_duration * sample_rate)

            if end_idx > len(t):
                end_idx = len(t)

            t_segment = t[start_idx:end_idx]
            if len(t_segment) == 0:
                continue

            note_wave = torch.zeros_like(t_segment)

            for h in range(1, n_harmonics + 1):
                amplitude = 0.6 / h  # Increased amplitude for counterpoint
                harmonic_wave = amplitude * torch.sin(
                    2 * torch.pi * freq * h * t_segment
                )
                note_wave += harmonic_wave

            envelope = torch.exp(-torch.linspace(0, 2, len(t_segment)))
            waveform[start_idx:end_idx] += note_wave * envelope

    else:
        raise ValueError(
            f"Invalid composition type: {composition_type}. Must be 1, 2, or 3."
        )

    # Normalize the waveform
    waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)

    return waveform


class SyntheticDataset(Dataset):
    """Dataset for synthetic music data"""

    def __init__(
        self,
        root: str = "data/synthetic",
        sample_rate: int = 16000,
        duration: float = 5.0,
        preprocessor: Optional[AudioPreprocessor] = None,
    ):
        self.root = root
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples_audio = int(sample_rate * duration)
        self.preprocessor = preprocessor or AudioPreprocessor(sample_rate=sample_rate)

        # Find all .wav files in the root directory
        self.audio_files = []
        if os.path.exists(root):
            for file in os.listdir(root):
                if file.endswith(".wav"):
                    self.audio_files.append(os.path.join(root, file))

        if len(self.audio_files) == 0:
            raise ValueError(
                f"No .wav files found in synthetic dataset at {root}. Please generate synthetic data first."
            )

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return Mel-Spectrogram and audio"""
        audio_path = self.audio_files[idx]

        # Load audio
        audio, sr = sf.read(audio_path)
        if sr != self.sample_rate:
            # Resample if necessary (simple implementation)
            audio = (
                audio[:: sr // self.sample_rate]
                if sr > self.sample_rate
                else np.repeat(audio, self.sample_rate // sr)
            )

        audio = audio[: self.num_samples_audio]  # Truncate to duration
        if len(audio) < self.num_samples_audio:
            # Pad if too short
            audio = np.pad(audio, (0, self.num_samples_audio - len(audio)))

        # Convert to tensor
        audio = torch.from_numpy(audio).float().unsqueeze(0)  # [1, samples]

        # Convert to Mel-Spectrogram
        mel_spec = self.preprocessor.to_mel_spectrogram(audio)

        # Resize to fixed size
        target_frames = 256
        if mel_spec.shape[-1] < target_frames:
            mel_spec = F.pad(mel_spec, (0, target_frames - mel_spec.shape[-1]))
        else:
            mel_spec = mel_spec[..., :target_frames]

        # Normalize
        mel_spec, _, _ = self.preprocessor.normalize(mel_spec)
        mel_spec = mel_spec.squeeze(0) if mel_spec.shape[0] == 1 else mel_spec[0]

        return mel_spec, audio.squeeze(0)


class ResidualBlock(nn.Module):
    """Residual block for better gradient flow"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class AudioAutoencoder(nn.Module):
    """Autoencoder for audio reconstruction in Mel-Spectrogram domain"""

    def __init__(
        self, n_mels: int = 128, latent_dim: int = 128
    ):  # Increased default latent_dim
        super().__init__()
        self.n_mels = n_mels
        self.latent_dim = latent_dim

        # Encoder with residual blocks
        self.encoder = nn.Sequential(
            nn.Conv1d(n_mels, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            ResidualBlock(256),
            nn.Conv1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            ResidualBlock(128),
            nn.Conv1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ResidualBlock(64),
            nn.Conv1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            ResidualBlock(32),
        )

        # Bottleneck
        self.fc_encode = nn.Linear(32 * 16, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 32 * 16)

        # Decoder with residual blocks
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ResidualBlock(64),
            nn.ConvTranspose1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            ResidualBlock(128),
            nn.ConvTranspose1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            ResidualBlock(256),
            nn.ConvTranspose1d(256, n_mels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode Mel-Spectrogram to latent space"""
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        z = self.fc_encode(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to Mel-Spectrogram"""
        x = self.fc_decode(z)
        x = x.view(x.shape[0], 32, 16)
        x = self.decoder(x)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Mel-Spectrogram [batch_size, n_mels, time_steps]
        Returns:
            reconstructed: Reconstructed Mel-Spectrogram
            latent: Representation in latent space
        """
        z = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed, z


class AudioTrainer:
    """Trainer for the audio Autoencoder"""

    def __init__(
        self,
        model: AudioAutoencoder,
        preprocessor: AudioPreprocessor,
        device: torch.device = torch.device("cpu"),
        learning_rate: float = 1e-3,
    ):
        self.model = model.to(device)
        self.preprocessor = preprocessor
        self.device = device
        self.optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )
        self.losses = []

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0

        for mel_specs, _ in tqdm(dataloader, desc="Training"):
            mel_specs = mel_specs.to(self.device)

            # Forward pass
            reconstructed, _ = self.model(mel_specs)

            # Calculate losses
            loss_mse = F.mse_loss(reconstructed, mel_specs)

            # Mel-spectrogram loss (compare in frequency domain)
            # Convert back to mel for loss (since reconstructed is in mel space)
            loss_mel = F.l1_loss(reconstructed, mel_specs)  # Simplified mel loss

            # Combined loss
            loss = loss_mse + 0.1 * loss_mel

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        self.losses.append(avg_loss)
        self.scheduler.step()

        return avg_loss

    @torch.no_grad()
    def reconstruct(self, mel_specs: torch.Tensor) -> torch.Tensor:
        """Reconstruir Mel-Spectrograms"""
        self.model.eval()
        reconstructed, _ = self.model(mel_specs)
        return reconstructed

    def compute_metrics(
        self, original: torch.Tensor, reconstructed: torch.Tensor
    ) -> dict:
        """
        Calculate quality metrics

        Args:
            original: Original Mel-Spectrogram [batch_size, n_mels, time_steps]
            reconstructed: Reconstructed Mel-Spectrogram
        Returns:
            dict with metrics
        """
        # MSE
        mse = F.mse_loss(reconstructed, original).item()

        # MAE
        mae = F.l1_loss(reconstructed, original).item()

        # Correlação média
        corr = (
            torch.nn.functional.cosine_similarity(
                original.view(original.shape[0], -1),
                reconstructed.view(reconstructed.shape[0], -1),
            )
            .mean()
            .item()
        )

        # Simplified PESQ (using correlation as proxy)
        pesq_score = corr

        return {
            "mse": mse,
            "mae": mae,
            "cosine_similarity": corr,
            "pesq_proxy": pesq_score,
        }

    def save_checkpoint(self, filepath: str):
        """Save model weights"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "losses": self.losses,
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved at: {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.losses = checkpoint.get("losses", [])
        print(f"Checkpoint loaded from: {filepath}")


def visualize_spectrograms(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    title: str = "Spectrogram Comparison",
    save_path: Optional[str] = None,
):
    """Visualize original vs reconstructed spectrograms"""
    fig, axes = plt.subplots(2, min(4, original.shape[0]), figsize=(16, 6))

    if original.shape[0] == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(title, fontsize=14)

    for idx in range(min(4, original.shape[0])):
        orig = original[idx].detach().cpu().numpy()
        recon = reconstructed[idx].detach().cpu().numpy()

        # Original
        im1 = axes[0, idx].imshow(orig, aspect="auto", cmap="viridis", origin="lower")
        axes[0, idx].set_title(f"Original {idx+1}")
        axes[0, idx].set_ylabel("Mel Bins")
        plt.colorbar(im1, ax=axes[0, idx])

        # Reconstructed
        im2 = axes[1, idx].imshow(recon, aspect="auto", cmap="viridis", origin="lower")
        axes[1, idx].set_title(f"Reconstructed {idx+1}")
        axes[1, idx].set_ylabel("Mel Bins")
        axes[1, idx].set_xlabel("Time")
        plt.colorbar(im2, ax=axes[1, idx])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"Imagem salva em: {save_path}")

    plt.show()


def plot_training_curves(
    losses: List[float],
    metrics_history: List[dict] = None,
    save_path: Optional[str] = None,
):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2 if metrics_history else 1, figsize=(14, 5))

    if metrics_history is None:
        axes = [axes]

    # Perda
    axes[0].plot(losses, marker="o", linewidth=2)
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Perda (MSE)")
    axes[0].set_title("Evolução da Perda de Treinamento")
    axes[0].grid(True)

    # Métricas
    if metrics_history:
        mse_vals = [m["mse"] for m in metrics_history]
        mae_vals = [m["mae"] for m in metrics_history]
        corr_vals = [m["cosine_similarity"] for m in metrics_history]

        axes[1].plot(mse_vals, marker="o", label="MSE", linewidth=2)
        axes[1].plot(mae_vals, marker="s", label="MAE", linewidth=2)
        axes[1].plot(corr_vals, marker="^", label="Similarity de Cosseno", linewidth=2)
        axes[1].set_xlabel("Época")
        axes[1].set_ylabel("Métrica")
        axes[1].set_title("Métricas de Validação")
        axes[1].legend()
        axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"Chart saved to: {save_path}")

    plt.show()


def export_audio(
    mel_spec: torch.Tensor,
    preprocessor: AudioPreprocessor,
    filepath: str,
    sample_rate: int = 16000,
):
    """Export Mel-Spectrogram to audio file"""
    # mel_spec is already normalized from the model
    mel_spec_denorm = mel_spec

    # Convert to audio
    waveform = preprocessor.from_mel_spectrogram(mel_spec_denorm.unsqueeze(0))

    # Normalize audio
    waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)

    # Save
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    sf.write(filepath, waveform.squeeze().detach().cpu().numpy(), sample_rate)
    print(f"Audio saved at: {filepath}")


def main():
    """Main function to train and test the model"""
    parser = argparse.ArgumentParser(
        description="Autoencoder for Synthetic Music Audio Regeneration"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "infer", "generate_synthetic"],
        default="train",
        help="Mode: train to train, infer to reconstruct, generate_synthetic to create synthetic music",
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/audio_autoencoder.pt",
        help="Path to save/load checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cpu or cuda)",
    )
    parser.add_argument("--n_mels", type=int, default=128, help="Number of Mel bins")
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=128,
        help="Latent space dimension",  # Updated default
    )
    parser.add_argument(
        "--synthetic_composition",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Type of synthetic music composition (1-3)",
    )
    parser.add_argument(
        "--synthetic_duration",
        type=float,
        default=10.0,
        help="Duration of synthetic music in seconds",
    )
    parser.add_argument(
        "--synthetic_harmonics",
        type=int,
        default=5,
        help="Number of harmonics for synthetic music generation",
    )

    args = parser.parse_args()

    # Configure device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create preprocessor
    preprocessor = AudioPreprocessor(sample_rate=16000, n_mels=args.n_mels)

    if args.mode == "train":
        print(f"\n{'='*50}")
        print("STARTING TRAINING")
        print(f"{'='*50}\n")

        # Create dataset
        dataset = SyntheticDataset(
            root="data/synthetic",
            sample_rate=16000,
            duration=5.0,
            preprocessor=preprocessor,
        )

        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        # Create model
        model = AudioAutoencoder(n_mels=args.n_mels, latent_dim=args.latent_dim)
        trainer = AudioTrainer(
            model, preprocessor, device=device, learning_rate=args.learning_rate
        )

        metrics_history = []

        for epoch in range(1, args.epochs + 1):
            loss = trainer.train_epoch(dataloader)
            print(f"Epoch {epoch}/{args.epochs} - Loss: {loss:.6f}")

            # Calculate metrics every 5 epochs
            if epoch % 5 == 0:
                sample_mel_specs, _ = next(iter(dataloader))
                sample_mel_specs = sample_mel_specs.to(device)
                reconstructed, _ = trainer.model(sample_mel_specs)
                metrics = trainer.compute_metrics(sample_mel_specs, reconstructed)
                metrics_history.append(metrics)
                print(
                    f"  MSE: {metrics['mse']:.6f}, "
                    f"MAE: {metrics['mae']:.6f}, "
                    f"Similarity: {metrics['cosine_similarity']:.4f}"
                )

        # Save checkpoint
        trainer.save_checkpoint(args.checkpoint)

        # Plot training curves
        plot_training_curves(
            trainer.losses, metrics_history, save_path="results/training_curves.png"
        )

        # Compare spectrograms
        print("\nGenerating spectrogram comparison...")
        sample_mel_specs, sample_audio = next(iter(dataloader))
        sample_mel_specs = sample_mel_specs.to(device)
        reconstructed, _ = trainer.model(sample_mel_specs)

        visualize_spectrograms(
            sample_mel_specs[:4],
            reconstructed[:4],
            title="Spectrogram Comparison: Original vs Reconstructed",
            save_path="results/spectrogram_comparison.png",
        )

        # Calculate final metrics
        final_metrics = trainer.compute_metrics(sample_mel_specs, reconstructed)
        print(f"\nFinal Metrics:")
        for key, value in final_metrics.items():
            print(f"  {key}: {value:.6f}")

    elif args.mode == "infer":
        print(f"\n{'='*50}")
        print("INFERENCE MODE - Reconstructing Audio")
        print(f"{'='*50}\n")

        # Create dataset
        dataset = SyntheticDataset(
            root="data/synthetic",
            sample_rate=16000,
            duration=5.0,
            preprocessor=preprocessor,
        )

        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        # Create model
        model = AudioAutoencoder(n_mels=args.n_mels, latent_dim=args.latent_dim)
        trainer = AudioTrainer(
            model, preprocessor, device=device, learning_rate=args.learning_rate
        )

        # Load checkpoint
        if os.path.exists(args.checkpoint):
            trainer.load_checkpoint(args.checkpoint)
        else:
            print(f"Warning: Checkpoint not found at {args.checkpoint}")
            print("Using untrained model for demonstration...")

        # Get samples
        sample_mel_specs, sample_audio = next(iter(dataloader))
        sample_mel_specs = sample_mel_specs.to(device)

        # Reconstruct
        reconstructed, latent = trainer.model(sample_mel_specs[:4])

        # Visualize
        visualize_spectrograms(
            sample_mel_specs[:4],
            reconstructed,
            title="Audio Reconstruction",
            save_path="results/inference_comparison.png",
        )

        # Calculate and display metrics
        metrics = trainer.compute_metrics(sample_mel_specs[:4], reconstructed)
        print("\nReconstruction Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")

        # Export audios (demonstration)
        print("\nExporting reconstructed audios...")
        for idx in range(min(2, sample_mel_specs.shape[0])):
            export_audio(
                reconstructed[idx],
                preprocessor,
                f"results/audio_reconstructed_{idx}.wav",
            )
            print(f"Audio {idx} exported")

    elif args.mode == "generate_synthetic":
        print(f"\n{'='*50}")
        print("SYNTHETIC MUSIC GENERATION MODE")
        print(f"{'='*50}\n")

        print(f"Generating synthetic music composition {args.synthetic_composition}")
        print(f"Duration: {args.synthetic_duration} seconds")
        print(f"Number of harmonics: {args.synthetic_harmonics}")

        # Generate synthetic music
        synthetic_waveform = generate_synthetic_music(
            sample_rate=16000,
            duration=args.synthetic_duration,
            composition_type=args.synthetic_composition,
            n_harmonics=args.synthetic_harmonics,
        )

        # Normalize the waveform to prevent clipping and ensure audibility
        synthetic_waveform = synthetic_waveform / (
            torch.max(torch.abs(synthetic_waveform)) + 1e-8
        )

        # Convert to Mel-Spectrogram for visualization
        mel_spec = preprocessor.to_mel_spectrogram(synthetic_waveform.unsqueeze(0))

        # Create data directory
        os.makedirs("data/synthetic", exist_ok=True)

        # Save the generated audio
        output_path = f"data/synthetic/synthetic_music_composition_{args.synthetic_composition}.wav"
        sf.write(output_path, synthetic_waveform.numpy(), 16000)
        print(f"Synthetic music saved to: {output_path}")

        # Visualize the Mel-Spectrogram
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.imshow(
            mel_spec[0].numpy(),
            aspect="auto",
            origin="lower",
            extent=[0, args.synthetic_duration, 0, preprocessor.n_mels],
        )
        plt.colorbar()
        plt.title(
            f"Synthetic Music Composition {args.synthetic_composition} - Mel-Spectrogram"
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Mel Bin")

        # Plot waveform
        plt.subplot(2, 1, 2)
        time_axis = torch.linspace(0, args.synthetic_duration, len(synthetic_waveform))
        plt.plot(time_axis.numpy(), synthetic_waveform.numpy())
        plt.title(
            f"Synthetic Music Composition {args.synthetic_composition} - Waveform"
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()

        spectrogram_path = f"results/synthetic_music_composition_{args.synthetic_composition}_analysis.png"
        plt.savefig(spectrogram_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Analysis plot saved to: {spectrogram_path}")

        print(f"\nSynthetic music generation completed!")
        print(f"Composition type: {args.synthetic_composition}")
        print(f"Duration: {args.synthetic_duration} seconds")
        print(f"Sample rate: 16000 Hz")


if __name__ == "__main__":
    main()
