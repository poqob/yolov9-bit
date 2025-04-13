#!/bin/bash
# YOLOv9 training script
# metrics for test computer: --batch 64 --img 640 --epochs 100 --close-mosaic 15

# Define variables
BATCH_SIZE=8
IMG_SIZE=640
MODEL_NAME="yolov9-t"
EPOCHS=150
CLOSE_MOSAIC=15

# Get custom name from command line if provided, otherwise use MODEL_NAME
RUN_NAME=${1:-$MODEL_NAME}

python train_dual.py --workers 8 --device 0 --batch $BATCH_SIZE \
  --data data/dataset.yaml --img $IMG_SIZE \
  --cfg models/detect/$MODEL_NAME.yaml --weights '' \
  --name $RUN_NAME --hyp hyp.scratch-high.yaml \
  --min-items 0 --epochs $EPOCHS --close-mosaic $CLOSE_MOSAIC