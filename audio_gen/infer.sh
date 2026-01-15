#!/bin/bash

# Audio Generation Inference Script
# This script reconstructs audio using the trained autoencoder model

# Set default values
CHECKPOINT=${CHECKPOINT:-../models/audio_autoencoder.pt}
MUSDB_ROOT=${MUSDB_ROOT:-data/MUSDB18}

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
echo "Starting audio generation inference..."
echo "Checkpoint: $CHECKPOINT, MUSDB Root: $MUSDB_ROOT, Device: $DEVICE"

python audio_gen.py \
    --mode infer \
    --checkpoint $CHECKPOINT \
    --musdb_root $MUSDB_ROOT \
    --device $DEVICE