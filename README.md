# ImageAudioGen

Complete generative artificial intelligence project that implements two deep learning models: image generation with Diffusion Models and audio regeneration with Autoencoders.

## Requirements and Environment Setup

This project requires Python 3.12 and uses a virtual environment for dependency management.

### Prerequisites
- Python 3.12
- pip (Python package installer)

### Setup Instructions

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd ImageAudioGen
   ```

2. **Create a virtual environment**:
   ```bash
   python3.12 -m venv venv
   ```

3. **Activate the virtual environment**:
   - On Linux/Mac:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify installation**:
   ```bash
   python --version  # Should show Python 3.12.x
   pip list  # Should show installed packages
   ```

## Project Description

This project evaluates applied AI technical capabilities, demonstrating:
- Mastery of deep learning frameworks (PyTorch)
- Knowledge of generative models (Diffusion Models and Autoencoders)
- Best practices in machine learning engineering
- Data preprocessing and evaluation

## Project Architecture

```
ImageAudioGen/
├── image_gen/
│   ├── image_gen.py        # Image generation with Diffusion Model
│   ├── train.sh            # Training script for image generation
│   └── infer.sh            # Inference script for image generation
├── audio_gen/
│   ├── audio_gen.py        # Audio regeneration with Autoencoder
│   ├── train.sh            # Training script for audio regeneration
│   └── infer.sh            # Inference script for audio regeneration
├── requirements.txt        # Dependencies for virtual environment
├── README.md               # This file
└── challenge.txt           # Requirements description
```

---

## Part 1: Image Generation (`image_gen/image_gen.py`)

### What it does?
Implements a complete **Diffusion Model** capable of generating synthetic images of handwritten digits (MNIST) from random Gaussian noise.

### Main Components

#### 1. **PositionalEncoding**
- Encodes temporal information (timesteps) of the diffusion
- Uses sine/cosine functions to create positional embeddings
- Allows the network to understand at which stage of the diffusion process it is

#### 2. **DiffusionModel (U-Net)**
- **Input**: Noisy image + timestep
- **Output**: Prediction of the added Gaussian noise
- **Architecture**: Simplified U-Net with:
  - Encoder: 2 convolution blocks + maxpooling (reduces dimensionality)
  - Decoder: 2 deconvolution blocks (restores original size)
  - Skip connections: Concatenate encoder and decoder features

#### 3. **DiffusionTrainer**
- **Forward Diffusion**: Progressively adds noise to the image (1000 timesteps)
- **Reverse Diffusion**: Iteratively removes noise to generate new images
- **Loss**: MSE between predicted noise and real noise

#### 4. **Metrics and Evaluation**
- **FID (Fréchet Inception Distance)**: Measures quality/diversity of generated images
- Visual comparison between epochs
- Training loss history

### How to Use

**Train the model:**
```bash
python image_gen.py --mode train \
    --epochs 20 \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --device cuda
```

**Generate samples with trained model:**
```bash
python image_gen.py --mode infer \
    --num_samples 16 \
    --checkpoint models/diffusion_model.pt \
    --device cuda
```

### Available Arguments

| Argument          | Type  | Default                   | Description                             |
| ----------------- | ----- | ------------------------- | --------------------------------------- |
| `--mode`          | str   | train                     | 'train' to train or 'infer' to generate |
| `--epochs`        | int   | 20                        | Number of training epochs               |
| `--batch_size`    | int   | 64                        | Batch size                              |
| `--learning_rate` | float | 1e-3                      | Learning rate                           |
| `--checkpoint`    | str   | models/diffusion_model.pt | Saved model path                        |
| `--num_samples`   | int   | 16                        | Number of images to generate            |
| `--device`        | str   | cuda/cpu                  | CPU or GPU                              |

### Generated Outputs

```
results/
├── training_loss.png           # Loss vs epoch graph
├── epoch_comparisons/
│   ├── samples_epoch_001.png   # Samples at epoch 1
│   ├── samples_epoch_005.png   # Samples at epoch 5
│   ├── samples_epoch_010.png   # Samples at epoch 10
│   ├── samples_epoch_020.png   # Samples at epoch 20
│   └── metrics_comparison.png  # FID vs Loss vs Epoch
└── final_samples.png           # Final generated samples

models/
└── diffusion_model.pt          # Trained model weights
```

### Expected Results

- **Quality**: Images get better as training increases
- **Diversity**: FID increases (higher standard deviation = more diversity)
- **Loss**: Decreases exponentially in the first epochs

---

## Part 2: Audio Regeneration (`audio_gen/audio_gen.py`)

### What it does?
Implements an **Autoencoder** to reconstruct audio stems from time-frequency domain representations (Mel-Spectrogram).

### Main Components

#### 1. **AudioPreprocessor**
- Converts audio to **Mel-Spectrogram** (frequency analysis)
- **Mel-Spectrogram**: Representation that mimics how the human ear perceives sound
  - Frequencies: Represented in logarithmic scale (Mel)
  - Y-axis: 128 Mel bins (standard)
  - X-axis: Time frames
- Uses **Griffin-Lim** to reconstruct audio from Mel-Spectrogram
- Normalizes data for training

#### 2. **SyntheticMUSDBDataset**
- Simulates MUSDB18 dataset (a standard in audio separation)
- Generates synthetic audio with:
  - Multiple harmonic frequencies (440Hz, 880Hz, 1320Hz, 1760Hz)
  - Variable amplitudes
  - Gaussian noise
- Resizes to fixed size (256 temporal frames)

#### 3. **AudioAutoencoder**
- **Input**: Mel-Spectrogram [128 mels × 256 timesteps]
- **Process**:
  1. Encoder (4 layers): Compresses to latent space (64 dimensions)
  2. Bottleneck: Compressed representation
  3. Decoder (4 layers): Reconstructs original Mel-Spectrogram
- **Output**: Reconstructed Mel-Spectrogram [128 × 256]

#### 4. **AudioTrainer**
- **Loss Function**: MSE between original and reconstructed Mel-Spectrograms
- **Optimizer**: Adam with learning rate scheduler
- **Gradient Clipping**: Prevents gradient explosion

#### 5. **Quality Metrics**

| Metric                | Description                       | Range               |
| --------------------- | --------------------------------- | ------------------- |
| **MSE**               | Mean squared error pixel by pixel | 0-∞ (lower=better)  |
| **MAE**               | Mean absolute error               | 0-∞ (lower=better)  |
| **Cosine Similarity** | Similarity between spectra        | 0-1 (higher=better) |
| **PESQ Proxy**        | Perceptual quality approximation  | 0-1 (higher=better) |

### How to Use

**Train the model:**
```bash
python audio_gen.py --mode train \
    --epochs 30 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --num_samples 100 \
    --n_mels 128 \
    --latent_dim 64 \
    --device cuda
```

**Reconstruct audio with trained model:**
```bash
python audio_gen.py --mode infer \
    --checkpoint models/audio_autoencoder.pt \
    --device cuda
```

### Available Arguments

| Argument          | Type  | Default                     | Description                                |
| ----------------- | ----- | --------------------------- | ------------------------------------------ |
| `--mode`          | str   | train                       | 'train' to train or 'infer' to reconstruct |
| `--epochs`        | int   | 30                          | Number of training epochs                  |
| `--batch_size`    | int   | 32                          | Batch size                                 |
| `--learning_rate` | float | 1e-3                        | Learning rate                              |
| `--num_samples`   | int   | 100                         | Number of dataset samples                  |
| `--checkpoint`    | str   | models/audio_autoencoder.pt | Saved model path                           |
| `--n_mels`        | int   | 128                         | Number of Mel bins                         |
| `--latent_dim`    | int   | 64                          | Latent space dimension                     |
| `--device`        | str   | cuda/cpu                    | CPU or GPU                                 |

### Generated Outputs

```
results/
├── training_curves.png                # Loss + Metrics vs Epoch
├── spectrogram_comparison.png         # Original vs Reconstructed
├── inference_comparison.png           # Test samples
├── audio_reconstructed_0.wav          # Reconstructed audio #0
└── audio_reconstructed_1.wav          # Reconstructed audio #1

models/
└── audio_autoencoder.pt               # Trained model weights
```

### Processing Flow

```
Original audio (16kHz, 5 seconds)
         ↓
   Mel-Spectrogram
   [1 × 128 × 256]
         ↓
    ENCODER
   (4 Conv1d layers)
         ↓
   Latent Space
   [1 × 64]
         ↓
    DECODER
   (4 ConvTranspose1d layers)
         ↓
   Reconstructed Mel-Spectrogram
   [1 × 128 × 256]
         ↓
Griffin-Lim Inverse
         ↓
Reconstructed Audio (16kHz)
```

### Expected Results

- **MSE**: Decreases during training (starts ~0.5, ends ~0.05)
- **Similarity**: Increases (starts ~0.5, ends ~0.95)
- **Perceptual Quality**: Reconstructed audio becomes increasingly faithful to the original

---

## How to Run

### Prerequisites

```bash
pip install torch torchaudio torchvision torchmetrics numpy matplotlib soundfile tqdm
```

### Complete Execution (Image + Audio)

You can run the models using either the provided shell scripts or direct Python commands:

#### Using Shell Scripts (Recommended)

```bash
# Step 1: Train image generation model
cd image_gen && ./train.sh

# Step 2: Generate new images
cd image_gen && ./infer.sh

# Step 3: Train audio regeneration model
cd audio_gen && ./train.sh

# Step 4: Reconstruct audio
cd audio_gen && ./infer.sh
```

#### Using Direct Python Commands

```bash
# Step 1: Train image generation model
python image_gen/image_gen.py --mode train --epochs 20 --batch_size 64

# Step 2: Generate new images
python image_gen/image_gen.py --mode infer --num_samples 16

# Step 3: Train audio regeneration model
python audio_gen/audio_gen.py --mode train --epochs 30 --batch_size 32

# Step 4: Reconstruct audio
python audio_gen/audio_gen.py --mode infer
```

#### Customizing Script Parameters

You can set environment variables to customize the scripts:

```bash
# For image generation training
cd image_gen
EPOCHS=10 BATCH_SIZE=32 ./train.sh

# For audio generation inference
cd audio_gen
CHECKPOINT=../models/my_model.pt ./infer.sh
```

---

## Metrics and Results

### Diffusion Model (Images)
- **Main Metric**: FID (Fréchet Inception Distance)
- **Visualization**: Sample comparison between epochs
- **Loss**: MSE between predicted and real noise

### Autoencoder (Audio)
- **Metrics**: MSE, MAE, Cosine Similarity, PESQ Proxy
- **Visualization**: Original vs reconstructed spectrograms
- **Analysis**: Convergence graphs

---

## Technical Highlights

✅ **Generative Models**: Diffusion Models (SOTA in image generation)  
✅ **Autoencoder Models**: Efficient compression and reconstruction  
✅ **Preprocessing**: Mel-Spectrogram for audio, Normalization for both  
✅ **Advanced Metrics**: FID, Cosine Similarity, PESQ Proxy  
✅ **Best Practices**: Checkpointing, Learning Rate Scheduling, Gradient Clipping  
✅ **Intuitive CLI**: Configurable arguments for easy experimentation  
✅ **Visualization**: Comparative graphs and quality analysis  

---

## Frameworks Used

- **PyTorch**: Main deep learning framework
- **Torchaudio**: Audio processing
- **Torchvision**: Image transformations
- **Matplotlib**: Visualization
- **Soundfile**: Audio export
- **Tqdm**: Progress bars

---

## References

- Ho et al. (2020): Denoising Diffusion Probabilistic Models (DDPM)
- Kingma & Welling (2013): Auto-Encoding Variational Bayes
- Mel-Frequency Cepstral Coefficients (MFCC) - Standard in audio processing

---

## Next Steps

- Implement VAE (Variational Autoencoder) for audio
- Add GAN for image generation
- Integration with real MUSDB18 data
- REST API for inference
- Web interface with Streamlit