#!/bin/bash
# YOLOv9 training script
# metrics for test computer: --batch 64 --img 640 --epochs 100 --close-mosaic 15
python train_dual.py --workers 8 --device 0 --batch 8 --data data/dataset.yaml --img 640 --cfg models/detect/yolov9-t.yaml --weights '' --name yolov9-t --hyp hyp.scratch-high.yaml --min-items 0 --epochs 150 --close-mosaic 15
