#!/bin/bash

# Audio Generation Inference Script
# This script reconstructs audio using the trained autoencoder model

# Set default values
CHECKPOINT=${CHECKPOINT:-models/audio_autoencoder.pt}

# Check for CUDA availability and set default device
DEFAULT_DEVICE="cpu"
DEVICE=${DEVICE:-$DEFAULT_DEVICE}

# Activate virtual environment
source ../venv/bin/activate

# Create results directory if it doesn't exist
mkdir -p ./results

# Run inference
echo "Starting audio generation inference..."
echo "Checkpoint: $CHECKPOINT, Device: $DEVICE"

python audio_gen.py \
    --mode infer \
    --checkpoint $CHECKPOINT \
    --device $DEVICE