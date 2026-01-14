#!/bin/bash

# Image Generation Inference Script
# This script generates images using the trained diffusion model

# Set default values
NUM_SAMPLES=${NUM_SAMPLES:-16}
CHECKPOINT=${CHECKPOINT:-../models/diffusion_model.pt}

# Check for CUDA availability and set default device
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    DEFAULT_DEVICE="cuda"
else
    DEFAULT_DEVICE="cpu"
fi
DEVICE=${DEVICE:-$DEFAULT_DEVICE}

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