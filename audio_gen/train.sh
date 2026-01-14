#!/bin/bash

# Audio Generation Training Script
# This script trains the autoencoder model for audio regeneration

# Set default values
EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-0.001}
NUM_SAMPLES=${NUM_SAMPLES:-100}
N_MELS=${N_MELS:-128}
LATENT_DIM=${LATENT_DIM:-64}
DEVICE=${DEVICE:-cuda}

# Activate virtual environment
source ..//venv/bin/activate

# Create models directory if it doesn't exist
mkdir -p ../models

# Run training
echo "Starting audio generation training..."
echo "Epochs: $EPOCHS, Batch Size: $BATCH_SIZE, Learning Rate: $LEARNING_RATE"
echo "Num Samples: $NUM_SAMPLES, N Mels: $N_MELS, Latent Dim: $LATENT_DIM, Device: $DEVICE"

python audio_gen.py \
    --mode train \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_samples $NUM_SAMPLES \
    --n_mels $N_MELS \
    --latent_dim $LATENT_DIM \
    --device $DEVICE