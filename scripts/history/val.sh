#!/bin/bash
# evaluate yolov9 models


# Define variables
BATCH_SIZE=8
IMG_SIZE=640
WEIGHTS_NAME=${1}




python val_dual.py --data data/dataset.yaml --img $IMG_SIZE --batch $BATCH_SIZE --conf 0.001 --iou 0.7 --device 0 --weights 'runs/train/'$WEIGHTS_NAME'/weights/best.pt' --save-json --name "$WEIGHTS_NAME""$IMG_SIZE"
