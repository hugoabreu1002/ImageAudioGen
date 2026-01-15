#!/bin/bash

# Audio Generation Training Script
# This script trains the autoencoder model for audio regeneration

# Set default values
EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-0.001}
N_MELS=${N_MELS:-128}
LATENT_DIM=${LATENT_DIM:-128}

# Check for CUDA availability and set default device
DEFAULT_DEVICE="cpu"
DEVICE=${DEVICE:-$DEFAULT_DEVICE}

# Activate virtual environment
source ../venv/bin/activate

# Set matplotlib backend to non-GUI
export MPLBACKEND=Agg

# Create models directory if it doesn't exist
mkdir -p models

# Check and generate synthetic dataset if not present
if [ ! -d "data/synthetic" ] || [ ! -f "data/synthetic/synthetic_music_composition_1.wav" ]; then
    echo "Synthetic dataset not found. Generating..."
    mkdir -p data/synthetic
    for COMPOSITION in {1..3}
    do
        echo "Generating composition $COMPOSITION..."
        if ! python audio_gen.py \
            --mode generate_synthetic \
            --synthetic_composition $COMPOSITION \
            --synthetic_duration 10.0 \
            --synthetic_harmonics 5 \
            --device $DEVICE; then
            echo "Failed to generate composition $COMPOSITION"
            exit 1
        fi
    done
    echo "Synthetic dataset generation completed."
    
    # Verify files were created
    if [ ! -f "data/synthetic/synthetic_music_composition_1.wav" ]; then
        echo "Error: Synthetic data generation failed - files not found"
        exit 1
    fi
else
    echo "Synthetic dataset found."
fi

# Run training
echo "Starting audio generation training..."
echo "Epochs: $EPOCHS, Batch Size: $BATCH_SIZE, Learning Rate: $LEARNING_RATE"
echo "N Mels: $N_MELS, Latent Dim: $LATENT_DIM, Device: $DEVICE"

python audio_gen.py \
    --mode train \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --n_mels $N_MELS \
    --latent_dim $LATENT_DIM \
    --device $DEVICE