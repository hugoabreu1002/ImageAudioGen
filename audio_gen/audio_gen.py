"""
Audio Regeneration with MUSDB18 using Autoencoder
Complete implementation to reconstruct audio stems

Requirements:
- Use MUSDB18 dataset or simulated
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

        # Transformação para Mel-Spectrogram
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=self.hop_length,
            normalized=True,
        )

        # Transformação inversa
        self.griffin_lim = T.GriffinLim(
            n_fft=n_fft, hop_length=self.hop_length, n_iter=32
        )

    def to_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert audio to Mel-Spectrogram

        Args:
            waveform: audio tensor [channels, samples]
        Returns:
            mel_spec: [channels, n_mels, time_steps]
        """
        # Se mono, adicionar dimensão de canal
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Converter para Mel-Spectrogram
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

        # Griffin-Lim for reconstruction
        waveform = self.griffin_lim(mel_spec)

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


class SyntheticMUSDBDataset(Dataset):
    """Synthetic dataset to simulate MUSDB18"""

    def __init__(
        self,
        num_samples: int = 100,
        sample_rate: int = 16000,
        duration: float = 5.0,
        preprocessor: Optional[AudioPreprocessor] = None,
    ):
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples_audio = int(sample_rate * duration)
        self.preprocessor = preprocessor or AudioPreprocessor(sample_rate=sample_rate)

    def generate_synthetic_audio(self, seed: int) -> torch.Tensor:
        """Generate synthetic audio with multiple frequencies"""
        torch.manual_seed(seed)
        np.random.seed(seed)

        t = torch.linspace(0, self.duration, self.num_samples_audio)

        # Generate signals with different frequencies and amplitudes
        freqs = torch.tensor(
            [440, 880, 1320, 1760], dtype=torch.float32
        )  # A, A5, E6, A6
        amps = torch.tensor([0.3, 0.2, 0.15, 0.1], dtype=torch.float32)

        audio = torch.zeros(self.num_samples_audio)
        for freq, amp in zip(freqs, amps):
            audio += amp * torch.sin(2 * np.pi * freq * t)

        # Add some noise
        audio += 0.05 * torch.randn_like(audio)

        # Normalize
        audio = audio / (torch.max(torch.abs(audio)) + 1e-8)

        return audio

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retornar amostra de áudio e seu Mel-Spectrogram"""
        # Gerar áudio sintético
        audio = self.generate_synthetic_audio(seed=idx)

        # Converter para Mel-Spectrogram
        mel_spec = self.preprocessor.to_mel_spectrogram(audio)

        # Redimensionar para tamanho fixo (crop ou pad)
        target_frames = 256
        if mel_spec.shape[-1] < target_frames:
            mel_spec = F.pad(mel_spec, (0, target_frames - mel_spec.shape[-1]))
        else:
            mel_spec = mel_spec[..., :target_frames]

        # Normalize
        mel_spec, _, _ = self.preprocessor.normalize(mel_spec)
        mel_spec = mel_spec.squeeze(0) if mel_spec.shape[0] == 1 else mel_spec[0]

        return mel_spec, audio


class AudioAutoencoder(nn.Module):
    """Autoencoder para reconstrução de áudio no domínio de Mel-Spectrogram"""

    def __init__(self, n_mels: int = 128, latent_dim: int = 64):
        super().__init__()
        self.n_mels = n_mels
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(n_mels, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )

        # Bottleneck
        self.fc_encode = nn.Linear(32 * 16, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 32 * 16)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.ConvTranspose1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.ConvTranspose1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.ConvTranspose1d(256, n_mels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Codificar Mel-Spectrogram para espaço latente"""
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        z = self.fc_encode(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodificar do espaço latente para Mel-Spectrogram"""
        x = self.fc_decode(z)
        x = x.view(x.shape[0], 32, 16)
        x = self.decoder(x)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Mel-Spectrogram [batch_size, n_mels, time_steps]
        Returns:
            reconstructed: Mel-Spectrogram reconstruído
            latent: Representação no espaço latente
        """
        z = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed, z


class AudioTrainer:
    """Treinador para o Autoencoder de áudio"""

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
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.5
        )
        self.losses = []

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Treinar por uma época"""
        self.model.train()
        total_loss = 0.0

        for mel_specs, _ in tqdm(dataloader, desc="Training"):
            mel_specs = mel_specs.to(self.device)

            # Forward pass
            reconstructed, _ = self.model(mel_specs)

            # Calcular loss
            loss = F.mse_loss(reconstructed, mel_specs)

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
        Calcular métricas de qualidade

        Args:
            original: Mel-Spectrogram original [batch_size, n_mels, time_steps]
            reconstructed: Mel-Spectrogram reconstruído
        Returns:
            dict com métricas
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

        # PESQ simplificado (usando correlação como proxy)
        pesq_score = corr

        return {
            "mse": mse,
            "mae": mae,
            "cosine_similarity": corr,
            "pesq_proxy": pesq_score,
        }

    def save_checkpoint(self, filepath: str):
        """Salvar pesos do modelo"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "losses": self.losses,
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        print(f"Checkpoint salvo em: {filepath}")

    def load_checkpoint(self, filepath: str):
        """Carregar pesos do modelo"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.losses = checkpoint.get("losses", [])
        print(f"Checkpoint carregado de: {filepath}")


def visualize_spectrograms(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    title: str = "Comparação de Espectrogramas",
    save_path: Optional[str] = None,
):
    """Visualizar espectrogramas originais vs reconstruídos"""
    fig, axes = plt.subplots(2, min(4, original.shape[0]), figsize=(16, 6))

    if original.shape[0] == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(title, fontsize=14)

    for idx in range(min(4, original.shape[0])):
        orig = original[idx].cpu().numpy()
        recon = reconstructed[idx].cpu().numpy()

        # Original
        im1 = axes[0, idx].imshow(orig, aspect="auto", cmap="viridis", origin="lower")
        axes[0, idx].set_title(f"Original {idx+1}")
        axes[0, idx].set_ylabel("Mel Bins")
        plt.colorbar(im1, ax=axes[0, idx])

        # Reconstruído
        im2 = axes[1, idx].imshow(recon, aspect="auto", cmap="viridis", origin="lower")
        axes[1, idx].set_title(f"Reconstruído {idx+1}")
        axes[1, idx].set_ylabel("Mel Bins")
        axes[1, idx].set_xlabel("Tempo")
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
    """Plotar curvas de treinamento"""
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
        axes[1].plot(
            corr_vals, marker="^", label="Similaridade de Cosseno", linewidth=2
        )
        axes[1].set_xlabel("Época")
        axes[1].set_ylabel("Métrica")
        axes[1].set_title("Métricas de Validação")
        axes[1].legend()
        axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"Gráfico salvo em: {save_path}")

    plt.show()


def export_audio(
    mel_spec: torch.Tensor,
    preprocessor: AudioPreprocessor,
    filepath: str,
    sample_rate: int = 16000,
):
    """Export Mel-Spectrogram to audio file"""
    # Denormalize
    mel_spec_denorm, _, _ = preprocessor.normalize(mel_spec)

    # Convert to audio
    waveform = preprocessor.from_mel_spectrogram(mel_spec_denorm.unsqueeze(0))

    # Normalize audio
    waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)

    # Salvar
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    sf.write(filepath, waveform.squeeze().cpu().numpy(), sample_rate)
    print(f"Áudio salvo em: {filepath}")


def main():
    """Função principal para treinar e testar o modelo"""
    parser = argparse.ArgumentParser(
        description="Autoencoder para Regeneração de Áudio MUSDB18"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "infer"],
        default="train",
        help="Modo: train para treinar, infer para reconstruir",
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Número de épocas de treinamento"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Tamanho do batch")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Taxa de aprendizado"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Número de amostras do dataset sintético",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/audio_autoencoder.pt",
        help="Caminho para salvar/carregar checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Dispositivo (cpu ou cuda)",
    )
    parser.add_argument("--n_mels", type=int, default=128, help="Número de Mel bins")
    parser.add_argument(
        "--latent_dim", type=int, default=64, help="Dimensão do espaço latente"
    )

    args = parser.parse_args()

    # Configurar dispositivo
    device = torch.device(args.device)
    print(f"Usando dispositivo: {device}")

    # Criar preprocessador
    preprocessor = AudioPreprocessor(sample_rate=16000, n_mels=args.n_mels)

    # Criar dataset
    dataset = SyntheticMUSDBDataset(
        num_samples=args.num_samples,
        sample_rate=16000,
        duration=5.0,
        preprocessor=preprocessor,
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Criar modelo
    model = AudioAutoencoder(n_mels=args.n_mels, latent_dim=args.latent_dim)
    trainer = AudioTrainer(
        model, preprocessor, device=device, learning_rate=args.learning_rate
    )

    if args.mode == "train":
        print(f"\n{'='*50}")
        print("INICIANDO TREINAMENTO")
        print(f"{'='*50}\n")

        metrics_history = []

        for epoch in range(1, args.epochs + 1):
            loss = trainer.train_epoch(dataloader)
            print(f"Época {epoch}/{args.epochs} - Loss: {loss:.6f}")

            # Calcular métricas a cada 5 épocas
            if epoch % 5 == 0:
                sample_mel_specs, _ = next(iter(dataloader))
                sample_mel_specs = sample_mel_specs.to(device)
                reconstructed, _ = trainer.model(sample_mel_specs)
                metrics = trainer.compute_metrics(sample_mel_specs, reconstructed)
                metrics_history.append(metrics)
                print(
                    f"  MSE: {metrics['mse']:.6f}, "
                    f"MAE: {metrics['mae']:.6f}, "
                    f"Similaridade: {metrics['cosine_similarity']:.4f}"
                )

        # Salvar checkpoint
        trainer.save_checkpoint(args.checkpoint)

        # Plotar curvas de treinamento
        plot_training_curves(
            trainer.losses, metrics_history, save_path="results/training_curves.png"
        )

        # Comparar espectrogramas
        print("\nGerando comparação de espectrogramas...")
        sample_mel_specs, sample_audio = next(iter(dataloader))
        sample_mel_specs = sample_mel_specs.to(device)
        reconstructed, _ = trainer.model(sample_mel_specs)

        visualize_spectrograms(
            sample_mel_specs[:4],
            reconstructed[:4],
            title="Comparação de Espectrogramas: Original vs Reconstruído",
            save_path="results/spectrogram_comparison.png",
        )

        # Calcular métricas finais
        final_metrics = trainer.compute_metrics(sample_mel_specs, reconstructed)
        print(f"\nMétricas Finais:")
        for key, value in final_metrics.items():
            print(f"  {key}: {value:.6f}")

    elif args.mode == "infer":
        print(f"\n{'='*50}")
        print("MODO INFERÊNCIA - Reconstruindo Áudio")
        print(f"{'='*50}\n")

        # Carregar checkpoint
        if os.path.exists(args.checkpoint):
            trainer.load_checkpoint(args.checkpoint)
        else:
            print(f"Aviso: Checkpoint não encontrado em {args.checkpoint}")
            print("Usando modelo não treinado para demonstração...")

        # Obter amostras
        sample_mel_specs, sample_audio = next(iter(dataloader))
        sample_mel_specs = sample_mel_specs.to(device)

        # Reconstruir
        reconstructed, latent = trainer.model(sample_mel_specs[:4])

        # Visualizar
        visualize_spectrograms(
            sample_mel_specs[:4],
            reconstructed,
            title="Reconstrução de Áudio",
            save_path="results/inference_comparison.png",
        )

        # Calcular e exibir métricas
        metrics = trainer.compute_metrics(sample_mel_specs[:4], reconstructed)
        print("\nMétricas de Reconstrução:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")

        # Exportar áudios (demonstração)
        print("\nExportando áudios reconstruídos...")
        for idx in range(min(2, sample_mel_specs.shape[0])):
            export_audio(
                reconstructed[idx],
                preprocessor,
                f"results/audio_reconstructed_{idx}.wav",
            )
            print(f"Áudio {idx} exportado")


if __name__ == "__main__":
    main()
