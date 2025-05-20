#!/bin/bash

# Set default values
WORKERS=${1:-8}    # Default: 8 workers
BATCH=${2:-8}      # Default: 8 batch size
IMG_SIZE=${3:-640} # Default: 640 image size
EPOCHS=${4:-100}   # Default: 100 epochs
ACTIVATION=${7:-"silu"}  # Default: Silu activation function
OPTIMIZER=${8:-"SGD"}   # Default: SGD optimizer
MODEL_DIR="dev/models/detect"

# Display the parameters being used
echo "Starting training with parameters:"
echo "  Workers: $WORKERS"
echo "  Batch size: $BATCH"
echo "  Image size: $IMG_SIZE"
echo "  Epochs: $EPOCHS"


# Create a log directory if it doesn't exist
LOG_DIR="logs/training"
mkdir -p $LOG_DIR

# Get the current timestamp for naming log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Find all YAML files in the model directory
YAML_FILES=$(find ${MODEL_DIR} -name "*.yaml" -type f)

# Run training for all models
echo "Training models from ${MODEL_DIR}..."
for MODEL in $YAML_FILES; do
    MODEL_NAME=$(basename $MODEL .yaml)
    echo "==============================================="
    echo "Starting training for $MODEL_NAME"
    echo "==============================================="
    LOG_FILE="$LOG_DIR/${MODEL_NAME}_${TIMESTAMP}.log"
    # Run training and log output
    python train_dual.py \
        --workers $WORKERS \
        --device 0 \
        --batch $BATCH \
        --data data/dataset.yaml \
        --img $IMG_SIZE \
        --cfg $MODEL \
        --weights '' \
        --name $MODEL_NAME \
        --hyp hyp.scratch-high.yaml \
        --min-items 0 \
        --epochs $EPOCHS \
        --activation $ACTIVATION \
        --optimizer $OPTIMIZER \
        --close-mosaic 15 2>&1 | tee $LOG_FILE

    echo "Training for $MODEL_NAME completed. Log saved to $LOG_FILE"
    echo ""
done

echo "All training jobs completed!"