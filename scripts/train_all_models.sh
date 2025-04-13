#!/bin/bash
# filepath: /mnt/newdisk/dosyalar/Dosyalar/projeler/py/EKG-1005-TUBITAK/scripts/train_all_models.sh
# YOLOv9 and GELAN training script with configurable parameters
# Usage: ./train_all_models.sh [workers] [batch] [img_size] [epochs]
# Example: ./train_all_models.sh 8 16 640 150

# Set default values
WORKERS=${1:-8}    # Default: 8 workers
BATCH=${2:-16}      # Default: 8 batch size
IMG_SIZE=${3:-640} # Default: 640 image size
EPOCHS=${4:-100}   # Default: 100 epochs

# Display the parameters being used
echo "Starting training with parameters:"
echo "  Workers: $WORKERS"
echo "  Batch size: $BATCH"
echo "  Image size: $IMG_SIZE"
echo "  Epochs: $EPOCHS"

# Directory where model configs are stored
MODEL_DIR="models/detect"

# Create a log directory if it doesn't exist
LOG_DIR="logs/training"
mkdir -p $LOG_DIR

# Get the current timestamp for naming log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Run training for all YOLOv9 models
echo "Training YOLOv9 models..."
for MODEL in $MODEL_DIR/yolov9*.yaml; do
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
        --close-mosaic 15 2>&1 | tee $LOG_FILE
    
    echo "Training for $MODEL_NAME completed. Log saved to $LOG_FILE"
    echo ""
done

echo "All training jobs completed!"