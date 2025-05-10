#!/bin/bash
# YOLOv9 training script
# metrics for test computer: --batch 64 --img 640 --epochs 100 --close-mosaic 15

# Define variables
BATCH_SIZE=4
IMG_SIZE=640
DEFAULT_MODEL_NAME="yolov9-t-residual"
MODEL_NAME=${1:-$DEFAULT_MODEL_NAME}  # Get model name from command line or use default
EPOCHS=100
CLOSE_MOSAIC=15
ACTIVATION=${7:-"silu"}  # Default: Silu activation function
OPTIMIZER=${8:-"SGD"}   # Default: SGD optimizer,CustomRMSpropOptimizer
CONFIG_FILE="dev/models/detect/$MODEL_NAME.yaml"

python train_dual.py --workers 8 --device 0 --batch $BATCH_SIZE \
  --data data/dataset.yaml --img $IMG_SIZE \
  --cfg $CONFIG_FILE --weights '' \
  --name $MODEL_NAME --hyp hyp.scratch-high.yaml \
  --min-items 0 --epochs $EPOCHS --close-mosaic $CLOSE_MOSAIC \
  --activation $ACTIVATION \
  --optimizer $OPTIMIZER \