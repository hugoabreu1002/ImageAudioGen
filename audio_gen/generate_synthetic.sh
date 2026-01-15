#!/bin/bash

# Synthetic Music Generation Script
# This script generates synthetic music using harmonic compositions

# Set default values
DURATION=${DURATION:-10.0}
HARMONICS=${HARMONICS:-5}

# Check for CUDA availability and set default device
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    DEFAULT_DEVICE="cuda"
else
    DEFAULT_DEVICE="cpu"
fi
DEVICE=${DEVICE:-$DEFAULT_DEVICE}

# Activate virtual environment
source ../venv/bin/activate

# Create data directory if it doesn't exist
mkdir -p ./data/synthetic

# Run synthetic music generation for compositions 1-3
echo "Starting synthetic music generation..."
echo "Duration: $DURATION seconds, Harmonics: $HARMONICS, Device: $DEVICE"

for COMPOSITION in {1..3}
do
    echo "Generating composition $COMPOSITION..."
    python audio_gen.py \
        --mode generate_synthetic \
        --synthetic_composition $COMPOSITION \
        --synthetic_duration $DURATION \
        --synthetic_harmonics $HARMONICS \
        --device $DEVICE
done

echo "Synthetic music generation completed!"