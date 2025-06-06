#!/bin/bash

# Model adı ve girdi dosyası doğrudan tanımlanmış
MODEL_NAME="custom5"
INPUT_FILE="data/val/10.png"

# Tahmin işlemini çalıştır
python detect_dual.py --weights runs/train/$MODEL_NAME/weights/best.pt --source $INPUT_FILE --conf-thres 0.25 --save-txt --save-conf
