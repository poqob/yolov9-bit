#!/bin/bash
# evaluate yolov9 models
python val_dual.py --data data/dataset.yaml --img 640 --batch 16 --conf 0.001 --iou 0.7 --device 0 --weights 'runs/train/yolov9-t/weights/best.pt' --save-json --name yolov9_t_640_val
