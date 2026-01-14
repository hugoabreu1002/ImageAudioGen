#!/bin/bash

# Image Generation Training Script
# This script trains the diffusion model for image generation

# Set default values
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-64}
LEARNING_RATE=${LEARNING_RATE:-0.001}
DEVICE=${DEVICE:-cuda}

# Activate virtual environment
source ../venv/bin/activate

# Create models directory if it doesn't exist
mkdir -p ../models

# Run training
echo "Starting image generation training..."
echo "Epochs: $EPOCHS, Batch Size: $BATCH_SIZE, Learning Rate: $LEARNING_RATE, Device: $DEVICE"

python image_gen.py \
    --mode train \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --device $DEVICE