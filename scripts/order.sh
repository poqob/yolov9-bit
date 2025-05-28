#!/bin/bash
# Comprehensive training script for all combinations
# Parameters:
# activation: selu, h_swish, elu, sinlu, sinlu_pozitive, sinlu_pozitive0.5
# optimizer: SGD, Adam, LION
# model: implementation-residual, implementation-t-cbam, implementation-t-mbconv, implementation-t
# batch: 8
# img: 640
# epochs: 100
# close-mosaic: 15

# Default is to start from the beginning
CONTINUE_TRAINING=false

# Check for command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --continue)
      CONTINUE_TRAINING=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--continue]"
      exit 1
      ;;
  esac
done

# Define arrays for models, activations, and optimizers
models=("implementation-residual" "implementation-t-cbam" "implementation-t-mbconv" "yolov9-t")
activations=("silu")
optimizers=("SGD" "Adam" "LION")

# Create log directory if it doesn't exist
mkdir -p logs/training_runs
LOG_FILE="logs/training_runs/training_sequence.log"

# Function to log the status of each run
log_status() {
    echo "$(date) - Starting: Model=$1, Activation=$2, Optimizer=$3" >> "$LOG_FILE"
}

# Function to log completion of a run
log_completion() {
    echo "$(date) - Completed: Model=$1, Activation=$2, Optimizer=$3" >> "$LOG_FILE"
}

# Function to find the last completed training
find_last_completed_training() {
    if [ ! -f "$LOG_FILE" ]; then
        return 1  # Log file does not exist
    fi
    
    # Get the last completed entry
    last_completed=$(grep "Completed:" "$LOG_FILE" | tail -n 1)
    
    if [ -z "$last_completed" ]; then
        return 1  # No completed training found
    fi
    
    # Extract model, activation, and optimizer from the log entry
    last_model=$(echo "$last_completed" | grep -oP 'Model=\K[^,]+')
    last_activation=$(echo "$last_completed" | grep -oP 'Activation=\K[^,]+')
    last_optimizer=$(echo "$last_completed" | grep -oP 'Optimizer=\K[^,]+')
    
    # Set global variables to track where we need to resume from
    LAST_MODEL="$last_model"
    LAST_ACTIVATION="$last_activation"
    LAST_OPTIMIZER="$last_optimizer"
    
    return 0  # Success
}

# Variables to track resumed training
RESUME=false
LAST_MODEL=""
LAST_ACTIVATION=""
LAST_OPTIMIZER=""

# Check if we need to continue training
if [ "$CONTINUE_TRAINING" = true ]; then
    echo "Checking for previously completed training runs..."
    
    if find_last_completed_training; then
        echo "Resuming training after: Model=$LAST_MODEL, Activation=$LAST_ACTIVATION, Optimizer=$LAST_OPTIMIZER"
        RESUME=true
    else
        echo "No completed training found. Starting from the beginning."
    fi
fi

# Loop through all combinations
for model in "${models[@]}"; do
    # Skip until we find the last completed model (if resuming)
    if [ "$RESUME" = true ] && [ "$model" != "$LAST_MODEL" ]; then
        echo "Skipping model: $model (already completed)"
        continue
    fi
    
    for activation in "${activations[@]}"; do
        # Skip until we find the last completed activation (if resuming)
        if [ "$RESUME" = true ] && [ "$model" = "$LAST_MODEL" ] && [ "$activation" != "$LAST_ACTIVATION" ]; then
            echo "Skipping activation: $activation (already completed)"
            continue
        fi
        
        for optimizer in "${optimizers[@]}"; do
            # Skip until we find the last completed optimizer (if resuming)
            if [ "$RESUME" = true ] && [ "$model" = "$LAST_MODEL" ] && [ "$activation" = "$LAST_ACTIVATION" ] && [ "$optimizer" = "$LAST_OPTIMIZER" ]; then
                echo "Found last completed training point. Resuming from next combination."
                RESUME=false
                continue
            fi
            
            # Skip if still finding the resume point
            if [ "$RESUME" = true ]; then
                echo "Skipping: $model/$activation/$optimizer (already completed)"
                continue
            fi
            
            # Create a unique name for this training run
            run_name="${model}_${activation}_${optimizer}"
            
            # Log the start of this training run
            log_status "$model" "$activation" "$optimizer"
            
            echo "Starting training: $run_name"
            
            # Run the training script with the current combination
            ./scripts/train.sh "$model" "$run_name" "$activation" "$optimizer"
            
            # Log completion of this training run
            log_completion "$model" "$activation" "$optimizer"
            
            # Optional: Add a delay between runs if needed
            sleep 5
        done
    done
done

echo "All training combinations completed!"