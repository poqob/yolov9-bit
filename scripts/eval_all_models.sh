#!/bin/bash
# filepath: /mnt/newdisk/dosyalar/Dosyalar/projeler/py/EKG-1005-TUBITAK/scripts/eval_all_models.sh
# YOLOv9 model evaluation script
# Usage: ./eval_all_models.sh [batch] [img_size]
# Example: ./eval_all_models.sh 16 640

# Set default values
BATCH=${1:-16}      # Default: 16 batch size
IMG_SIZE=${2:-640}  # Default: 640 image size

# Display the parameters being used
echo "Starting evaluation with parameters:"
echo "  Batch size: $BATCH"
echo "  Image size: $IMG_SIZE"

# Create log directory if it doesn't exist
EVAL_LOG_DIR="logs/evaluation"
SUMMARY_DIR="logs/summary"
mkdir -p $EVAL_LOG_DIR
mkdir -p $SUMMARY_DIR

# Get the current timestamp for naming log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUMMARY_FILE="$SUMMARY_DIR/eval_summary_${TIMESTAMP}.txt"

# Create summary header
echo "YOLOv9 Model Evaluation Summary ($(date))" > $SUMMARY_FILE
echo "=========================================" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE
echo "Parameters:" >> $SUMMARY_FILE
echo "  - Batch size: $BATCH" >> $SUMMARY_FILE
echo "  - Image size: $IMG_SIZE" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

# Table header for summary
echo "| Model          | Dataset   | mAP@0.5:0.95 | mAP@0.5 | Precision | Recall | F1-Score |" >> $SUMMARY_FILE
echo "|----------------|-----------|--------------|---------|-----------|--------|----------|" >> $SUMMARY_FILE

# Function to evaluate a model on validation and test sets
evaluate_model() {
    local MODEL_PATH=$1
    local MODEL_NAME=$(basename $MODEL_PATH | sed 's/\.pt$//')
    
    echo "==============================================="
    echo "Evaluating model: $MODEL_NAME"
    echo "==============================================="
    
    # Log file for this model's evaluation
    EVAL_LOG_FILE="$EVAL_LOG_DIR/${MODEL_NAME}_${TIMESTAMP}.log"
    
    # Evaluate on validation set
    echo "Evaluating on validation set..."
    python val.py \
        --weights $MODEL_PATH \
        --data data/dataset.yaml \
        --batch $BATCH \
        --img $IMG_SIZE \
        --task val \
        --verbose \
        --save-txt \
        --save-conf \
        --conf-thres 0.25 \
        --iou-thres 0.6 \
        --device 0 2>&1 | tee -a $EVAL_LOG_FILE
    
    # Evaluate on test set
    echo "Evaluating on test set..."
    python val.py \
        --weights $MODEL_PATH \
        --data data/dataset.yaml \
        --batch $BATCH \
        --img $IMG_SIZE \
        --task test \
        --verbose \
        --save-txt \
        --save-conf \
        --conf-thres 0.25 \
        --iou-thres 0.6 \
        --device 0 2>&1 | tee -a $EVAL_LOG_FILE
    
    # Extract metrics from logs and add to summary
    # For validation set
    VAL_MAP=$(grep "all" $EVAL_LOG_FILE | grep "val" | awk '{print $12}' | head -1)
    VAL_MAP50=$(grep "all" $EVAL_LOG_FILE | grep "val" | awk '{print $10}' | head -1)
    VAL_PRECISION=$(grep "precision" $EVAL_LOG_FILE | grep "val" | awk '{print $4}' | head -1)
    VAL_RECALL=$(grep "recall" $EVAL_LOG_FILE | grep "val" | awk '{print $4}' | head -1)
    VAL_F1=$(grep "F1" $EVAL_LOG_FILE | grep "val" | awk '{print $3}' | head -1)
    
    # For test set
    TEST_MAP=$(grep "all" $EVAL_LOG_FILE | grep "test" | awk '{print $12}' | head -1)
    TEST_MAP50=$(grep "all" $EVAL_LOG_FILE | grep "test" | awk '{print $10}' | head -1)
    TEST_PRECISION=$(grep "precision" $EVAL_LOG_FILE | grep "test" | awk '{print $4}' | head -1)
    TEST_RECALL=$(grep "recall" $EVAL_LOG_FILE | grep "test" | awk '{print $4}' | head -1)
    TEST_F1=$(grep "F1" $EVAL_LOG_FILE | grep "test" | awk '{print $3}' | head -1)
    
    # Add to summary table
    echo "| $MODEL_NAME | Validation | ${VAL_MAP:-N/A} | ${VAL_MAP50:-N/A} | ${VAL_PRECISION:-N/A} | ${VAL_RECALL:-N/A} | ${VAL_F1:-N/A} |" >> $SUMMARY_FILE
    echo "| $MODEL_NAME | Test       | ${TEST_MAP:-N/A} | ${TEST_MAP50:-N/A} | ${TEST_PRECISION:-N/A} | ${TEST_RECALL:-N/A} | ${TEST_F1:-N/A} |" >> $SUMMARY_FILE
    
    echo "Evaluation for $MODEL_NAME completed. Log saved to $EVAL_LOG_FILE"
    echo ""
}

# Function to find all trained models
find_trained_models() {
    local MODELS_DIR="runs/train"
    
    # Check if the directory exists
    if [ ! -d "$MODELS_DIR" ]; then
        echo "Error: Training directory not found: $MODELS_DIR"
        exit 1
    fi
    
    # Find all best.pt and last.pt files in the model directories
    local MODEL_PATHS=()
    
    echo "Finding trained models..."
    
    # First priority: best.pt files
    for MODEL_PATH in $(find $MODELS_DIR -name "best.pt"); do
        MODEL_DIR=$(dirname $MODEL_PATH)
        MODEL_NAME=$(basename $MODEL_DIR)
        echo "Found model: $MODEL_NAME (best weights)"
        MODEL_PATHS+=("$MODEL_PATH")
    done
    
    echo "Found ${#MODEL_PATHS[@]} trained models."
    
    # Return the array of model paths
    echo "${MODEL_PATHS[@]}"
}

# Find all trained models
MODEL_PATHS=($(find_trained_models))

# Exit if no models found
if [ ${#MODEL_PATHS[@]} -eq 0 ]; then
    echo "No trained models found. Please train models first."
    exit 1
fi

# Evaluate each model
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    evaluate_model "$MODEL_PATH"
done

# Include class-wise performance for each model
echo "" >> $SUMMARY_FILE
echo "Class-wise Performance" >> $SUMMARY_FILE
echo "======================" >> $SUMMARY_FILE

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    MODEL_NAME=$(basename $(dirname $MODEL_PATH))
    EVAL_LOG_FILE="$EVAL_LOG_DIR/${MODEL_NAME}_${TIMESTAMP}.log"
    
    echo "" >> $SUMMARY_FILE
    echo "Model: $MODEL_NAME" >> $SUMMARY_FILE
    echo "-------------" >> $SUMMARY_FILE
    
    # Extract class-wise metrics from logs
    CLASS_LINES=$(grep -A 20 "Class" $EVAL_LOG_FILE | grep -v "Class" | grep -v "^$" | head -6)
    if [ ! -z "$CLASS_LINES" ]; then
        echo "| Class | Images | Labels | P | R | mAP@.5 | mAP@.5:.95 |" >> $SUMMARY_FILE
        echo "|-------|--------|--------|---|---|--------|------------|" >> $SUMMARY_FILE
        echo "$CLASS_LINES" | while read -r line; do
            if [[ $line == *"all"* ]]; then
                continue
            fi
            # Format the line for the markdown table
            CLASS_NAME=$(echo $line | awk '{print $1}')
            IMAGES=$(echo $line | awk '{print $2}')
            LABELS=$(echo $line | awk '{print $3}')
            P=$(echo $line | awk '{print $4}')
            R=$(echo $line | awk '{print $5}')
            MAP50=$(echo $line | awk '{print $6}')
            MAP=$(echo $line | awk '{print $7}')
            echo "| $CLASS_NAME | $IMAGES | $LABELS | $P | $R | $MAP50 | $MAP |" >> $SUMMARY_FILE
        done
    else
        echo "Class-wise metrics not available" >> $SUMMARY_FILE
    fi
done

echo ""
echo "Evaluation complete! Summary saved to $SUMMARY_FILE"
echo "Individual logs saved to $EVAL_LOG_DIR/"

# Generate visualization of results
echo "Generating visualizations of results..."
# Add to the Python visualization code section in eval_all_models.sh
python -c "
import matplotlib.pyplot as plt
import pandas as pd
import re
import os
import numpy as np
from sklearn.metrics import roc_curve, auc

# Read the summary file
with open('$SUMMARY_FILE', 'r') as f:
    content = f.read()

# Extract data
pattern = r'\| ([^ ]+) \| (Validation|Test) \| ([^ ]+) \| ([^ ]+) \| ([^ ]+) \| ([^ ]+) \| ([^ ]+) \|'
matches = re.findall(pattern, content)

# Create DataFrame
data = []
for match in matches:
    model, dataset, mAP, mAP50, precision, recall, f1 = match
    # Convert to numeric, handle N/A
    mAP = float(mAP) if mAP != 'N/A' else 0
    mAP50 = float(mAP50) if mAP50 != 'N/A' else 0
    precision = float(precision) if precision != 'N/A' else 0
    recall = float(recall) if recall != 'N/A' else 0
    f1 = float(f1) if f1 != 'N/A' else 0
    data.append([model, dataset, mAP, mAP50, precision, recall, f1])

df = pd.DataFrame(data, columns=['Model', 'Dataset', 'mAP', 'mAP50', 'Precision', 'Recall', 'F1'])

# Create directory for plots
os.makedirs('logs/plots', exist_ok=True)

# Plot mAP for each model
plt.figure(figsize=(12, 6))
val_data = df[df['Dataset'] == 'Validation']
test_data = df[df['Dataset'] == 'Test']

# Sort by validation mAP
sorted_models = val_data.sort_values('mAP', ascending=False)['Model'].values

# Reindex based on sorted models
val_data = val_data.set_index('Model').loc[sorted_models].reset_index()
test_data = test_data.set_index('Model').loc[sorted_models].reset_index()

x = range(len(sorted_models))
width = 0.35

plt.bar(x, val_data['mAP'], width, label='Validation')
plt.bar([i + width for i in x], test_data['mAP'], width, label='Test')

plt.xlabel('Model')
plt.ylabel('mAP@0.5:0.95')
plt.title('Model Performance Comparison')
plt.xticks([i + width/2 for i in x], sorted_models, rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('logs/plots/map_comparison_$TIMESTAMP.png')

# Plot precision-recall graph
plt.figure(figsize=(10, 6))
for i, model in enumerate(sorted_models):
    val_row = val_data[val_data['Model'] == model].iloc[0]
    plt.scatter(val_row['Recall'], val_row['Precision'], label=model)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision vs Recall for Different Models (Validation)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('logs/plots/precision_recall_$TIMESTAMP.png')

# Generate ROC curves using per-class confidence data from validation logs
plt.figure(figsize=(10, 8))

# Get list of class names
with open('data/classes.txt', 'r') as f:
    class_names = [line.strip() for line in f]

# Function to extract confidence scores and ground truths from detection output
def extract_detection_data(model_name):
    eval_log_file = f'$EVAL_LOG_DIR/{model_name}_$TIMESTAMP.log'
    
    # Find prediction files directory from the log
    log_content = open(eval_log_file, 'r').read()
    match = re.search(r'Results saved to (.*?)\\n', log_content)
    if not match:
        return None
        
    pred_dir = match.group(1).strip()
    pred_dir = pred_dir.replace('\033[1m', '').replace('\033[0m', '')  # Remove color codes
    
    # Get prediction files
    labels_dir = os.path.join(pred_dir, 'labels')
    if not os.path.exists(labels_dir):
        return None
        
    # Each class gets its own ROC curve
    all_scores = {}
    all_truths = {}
    
    for class_idx in range(len(class_names)):
        all_scores[class_idx] = []
        all_truths[class_idx] = []
    
    # Process each prediction file
    for pred_file in os.listdir(labels_dir):
        if not pred_file.endswith('.txt'):
            continue
            
        # Get corresponding ground truth file from val/labels
        gt_file = os.path.join('data/val/labels', pred_file)
        if not os.path.exists(gt_file):
            continue
            
        # Read predictions
        preds = {}
        try:
            with open(os.path.join(labels_dir, pred_file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        cls_id = int(parts[0])
                        conf = float(parts[-1]) if len(parts) >= 6 else 1.0  # Confidence is last column if saved
                        if cls_id not in preds or conf > preds[cls_id]:
                            preds[cls_id] = conf
        except:
            continue
            
        # Read ground truths
        gt_classes = set()
        try:
            with open(gt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        cls_id = int(parts[0])
                        gt_classes.add(cls_id)
        except:
            continue
            
        # For each class, record if it was present (truth=1) and the confidence score
        for cls_id in range(len(class_names)):
            # Add the highest confidence for this class, or 0 if not predicted
            all_scores[cls_id].append(preds.get(cls_id, 0))
            # Add whether this class was actually present
            all_truths[cls_id].append(1 if cls_id in gt_classes else 0)
    
    return all_scores, all_truths

# Plot micro-average ROC curve for each model (better for multi-class)
plt.figure(figsize=(10, 8))

for model in sorted_models:
    try:
        result = extract_detection_data(model)
        if result is None:
            continue
            
        all_scores, all_truths = result
        
        # Combine all classes for micro-average
        y_true = []
        y_score = []
        
        for cls_id in range(len(class_names)):
            y_true.extend(all_truths[cls_id])
            y_score.extend(all_scores[cls_id])
            
        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{model} (AUC = {roc_auc:.2f})')
    except Exception as e:
        print(f'Error generating ROC curve for {model}: {e}')
        continue

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('logs/plots/roc_curve_$TIMESTAMP.png')

# Also plot per-class ROC curves for the best model
best_model = sorted_models[0]
try:
    result = extract_detection_data(best_model)
    if result is not None:
        all_scores, all_truths = result
        
        plt.figure(figsize=(12, 10))
        
        for cls_id in range(len(class_names)):
            if sum(all_truths[cls_id]) > 0:  # Only plot if we have positive samples
                fpr, tpr, _ = roc_curve(all_truths[cls_id], all_scores[cls_id])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{class_names[cls_id]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Per-Class ROC Curves for {best_model}')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'logs/plots/roc_curve_per_class_{best_model}_{$TIMESTAMP}.png')
except Exception as e:
    print(f'Error generating per-class ROC curves: {e}')

print('Plots saved to logs/plots/')
" 2>/dev/null

echo "Visualizations including ROC curves saved to logs/plots/"