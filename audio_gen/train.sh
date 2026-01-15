#!/bin/bash

# Audio Generation Training Script
# This script trains the autoencoder model for audio regeneration

# Set default values
EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-0.001}
MUSDB_ROOT=${MUSDB_ROOT:-data/musdb18}
N_MELS=${N_MELS:-128}
LATENT_DIM=${LATENT_DIM:-128}

# Check for CUDA availability and set default device
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    DEFAULT_DEVICE="cuda"
else
    DEFAULT_DEVICE="cpu"
fi
DEVICE=${DEVICE:-$DEFAULT_DEVICE}

# Activate virtual environment
source ../venv/bin/activate

# Create models directory if it doesn't exist
mkdir -p models

# Check and download MUSDB18 dataset if not present
if [ ! -d "data/musdb18" ]; then
    echo "MUSDB18 dataset not found. Downloading..."
    mkdir -p data
    cd data
    
    # Download MUSDB18 dataset (~8GB)
    wget -O musdb18.zip https://zenodo.org/record/1117372/files/musdb18.zip
    
    # Extract the dataset
    unzip musdb18.zip
    
    # Clean up zip file
    rm musdb18.zip
    
    cd ..
    echo "MUSDB18 dataset downloaded and extracted successfully."
else
    echo "MUSDB18 dataset found at data/musdb18"
fi

# Run training
echo "Starting audio generation training..."
echo "Epochs: $EPOCHS, Batch Size: $BATCH_SIZE, Learning Rate: $LEARNING_RATE"
echo "MUSDB Root: $MUSDB_ROOT, N Mels: $N_MELS, Latent Dim: $LATENT_DIM, Device: $DEVICE"

python audio_gen.py \
    --mode train \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --musdb_root $MUSDB_ROOT \
    --n_mels $N_MELS \
    --latent_dim $LATENT_DIM \
    --device $DEVICE