#!/bin/bash

# Image Generation Inference Script
# This script generates images using the trained diffusion model

# Set default values
NUM_SAMPLES=${NUM_SAMPLES:-16}
CHECKPOINT=${CHECKPOINT:-../models/diffusion_model.pt}
DEVICE=${DEVICE:-cuda}

# Activate virtual environment
source ../venv/bin/activate

# Create results directory if it doesn't exist
mkdir -p ../results

# Run inference
echo "Starting image generation inference..."
echo "Num Samples: $NUM_SAMPLES, Checkpoint: $CHECKPOINT, Device: $DEVICE"

python image_gen.py \
    --mode infer \
    --num_samples $NUM_SAMPLES \
    --checkpoint $CHECKPOINT \
    --device $DEVICE