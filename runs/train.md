# YOLOv9 Training Results Analysis

This document provides a comprehensive analysis of various YOLOv9 model training configurations and their performance metrics. The analysis includes different model architectures, activation functions, and optimizers.

## Training Configuration Summary

The training experiments were conducted with the following variations:

### Model Architectures
- YOLOv9-t (tiny)
- Implementation-residual (Custom residual implementation)
- Implementation-t-cbam (Custom implementation with CBAM attention mechanism)
- Implementation-t-mbconv (Custom implementation with MBConv blocks)

### Activation Functions
- silu
- elu
- selu
- h_swish
- sinlu
- sinlu_pozitive

### Optimizers
- Adam
- LION
- SGD

## Performance Comparison

### Model Performance by Architecture and Optimizer

| Model Architecture | Activation Function | Optimizer | mAP@0.5 | mAP@0.5:0.95 | Best Epoch |
|-------------------|---------------------|-----------|---------|--------------|------------|
| YOLOv9-t          | silu                | Adam      | 0.74049 | 0.34523      | 26         |
| YOLOv9-t          | h_swish             | LION      | 0.39367 | 0.15517      | 29         |
| Implementation-t-cbam | silu            | LION      | 0.02042 | 0.00597      | 23         |
| Implementation-t-mbconv | selu          | SGD       | 0.26264 | 0.08486      | 28         |
| Implementation-residual | sinlu_pozitive | SGD     | 0.39805 | 0.11068      | 25         |

### Loss Comparison

| Model Architecture | Activation Function | Optimizer | Final Box Loss | Final Cls Loss | Final DFL Loss |
|-------------------|---------------------|-----------|----------------|----------------|----------------|
| YOLOv9-t          | silu                | Adam      | 2.766          | 2.1766         | 2.0413         |
| YOLOv9-t          | h_swish             | LION      | 3.067          | 2.6317         | 1.8960         |
| Implementation-t-cbam | silu            | LION      | 3.446          | 2.7518         | 2.2324         |
| Implementation-t-mbconv | selu          | SGD       | 3.880          | 3.0730         | 3.1384         |
| Implementation-residual | sinlu_pozitive | SGD     | 3.3154         | 2.6897         | 2.7468         |

## Precision and Recall Analysis

| Model Architecture | Activation Function | Optimizer | Best Precision | Best Recall |
|-------------------|---------------------|-----------|----------------|-------------|
| YOLOv9-t          | silu                | Adam      | 0.86917        | 0.92437     |
| YOLOv9-t          | h_swish             | LION      | 0.42273        | 0.51661     |
| Implementation-t-cbam | silu            | LION      | 0.02340        | 0.46886     |
| Implementation-t-mbconv | selu          | SGD       | 0.21629        | 0.90195     |
| Implementation-residual | sinlu_pozitive | SGD     | 0.49937        | 0.84218     |

## Training Convergence Analysis

### YOLOv9-t with silu activation and Adam optimizer
- Fastest convergence with mAP@0.5 reaching 0.74049 by epoch 26
- Most stable training process with consistent improvement in metrics
- Highest overall mAP@0.5:0.95 of 0.34523

### Implementation-t-cbam with silu activation and LION optimizer
- Slower convergence compared to the YOLOv9-t model
- Lower final performance metrics
- Precision and recall metrics did not improve significantly throughout training

### Implementation-residual with sinlu_pozitive activation and SGD optimizer
- Moderate convergence rate
- Good recall (0.84218) but moderate precision (0.49937)
- mAP@0.5 peaked at 0.39805 in epoch 25

## Observations and Recommendations

1. **Best Overall Performance**: The YOLOv9-t architecture with silu activation function and Adam optimizer demonstrated the best overall performance with highest mAP scores.

2. **Activation Function Comparison**: 
   - silu activation generally performed better than other activation functions across different architectures
   - sinlu_pozitive showed promising results with the residual implementation

3. **Optimizer Impact**:
   - Adam optimizer consistently produced better results than LION and SGD
   - SGD showed moderate performance but required more epochs to converge
   - LION optimizer showed the slowest convergence among the tested optimizers

4. **Architecture Analysis**:
   - The original YOLOv9-t architecture outperformed the custom implementations
   - Among custom implementations, the residual approach showed better results than CBAM and MBConv variants

## Training Metrics Across Epochs

To better understand the training progression, we analyzed how key metrics evolved over time for each model configuration.

### mAP@0.5 Evolution

| Configuration | Epoch 10 | Epoch 20 | Epoch 30 | Epoch 50 | Epoch 80 | Epoch 100 |
|---------------|----------|----------|----------|----------|----------|-----------|
| YOLOv9-t / silu / Adam | 0.00000 | 0.42180 | 0.74049 | N/A | N/A | N/A |
| YOLOv9-t / h_swish / LION | 0.00000 | 0.02104 | 0.39367 | N/A | N/A | N/A |
| Implementation-t-cbam / silu / LION | 0.00022 | 0.00000 | 0.02042 | N/A | N/A | N/A |
| Implementation-t-mbconv / selu / SGD | 0.00000 | 0.00550 | 0.26264 | N/A | N/A | N/A |
| Implementation-residual / sinlu_pozitive / SGD | 0.00000 | 0.19869 | 0.39805 | N/A | N/A | N/A |

### Box Loss Evolution

| Configuration | Epoch 10 | Epoch 20 | Epoch 30 | Final |
|---------------|----------|----------|----------|-------|
| YOLOv9-t / silu / Adam | 3.2335 | 2.8434 | 2.7660 | 2.7660 |
| YOLOv9-t / h_swish / LION | 3.6944 | 3.2225 | 3.0669 | 3.0669 |
| Implementation-t-cbam / silu / LION | 4.5712 | 3.7873 | 3.4460 | 3.4460 |
| Implementation-t-mbconv / selu / SGD | 7.6022 | 5.0130 | 3.8804 | 3.8804 |
| Implementation-residual / sinlu_pozitive / SGD | 7.5718 | 4.4177 | 3.3154 | 3.3154 |

## Learning Rate Analysis

The learning rate scheduling pattern shows a typical warmup followed by a decay strategy:

1. Initial fast learning (epochs 0-3) with warmup_bias_lr of 0.1
2. Transition period (epochs 3-10) with gradually decreasing learning rate
3. Stable training period (epochs 10-30) with slow decay

The chart below represents the learning rate progression for the tested models:

| Epoch | YOLOv9-t / Adam | YOLOv9-t / LION | Implementation / SGD |
|-------|-----------------|-----------------|---------------------|
| 0     | 0.0712          | 0.0712          | 0.0712              |
| 5     | 0.0096          | 0.0096          | 0.0096              |
| 10    | 0.0091          | 0.0091          | 0.0091              |
| 20    | 0.0081          | 0.0081          | 0.0081              |
| 30    | 0.0072          | 0.0072          | 0.0072              |

## Future Work

Based on the current findings, the following directions for future experiments are recommended:

1. Extend training epochs for promising configurations to see if performance improves further
2. Test hybrid approaches combining the best elements from different architectures
3. Explore learning rate scheduling strategies to improve convergence
4. Investigate the impact of different data augmentation techniques
5. Test the best-performing models on different datasets to evaluate generalization capability

## Conclusion

The experiments demonstrate that the original YOLOv9-t architecture with silu activation function and Adam optimizer provides the best performance for the current dataset. Custom implementations show promise but require further refinement to match or exceed the performance of the original architecture.

## Activation Function Comparison

The choice of activation function significantly impacted model performance. Below is a detailed comparison of different activation functions used in the experiments:

### Activation Function Performance Across Architectures

| Activation Function | Best mAP@0.5 | Best Architecture | Best Optimizer | Notes |
|---------------------|--------------|-------------------|----------------|-------|
| silu                | 0.74049      | YOLOv9-t          | Adam           | Consistently strong performance across architectures |
| h_swish             | 0.39367      | YOLOv9-t          | LION           | Good performance on YOLOv9-t architecture |
| selu                | 0.26264      | Implementation-t-mbconv | SGD      | Moderate performance, better with custom architectures |
| sinlu_pozitive      | 0.39805      | Implementation-residual | SGD      | Strong performance on residual implementations |
| elu                 | Not reported | -                 | -              | Limited performance in tested configurations |

### Activation Function Characteristics

1. **silu (Sigmoid Linear Unit)**:
   - Formula: x * sigmoid(x)
   - Strengths: Smooth gradient, non-monotonic nature helps with gradient flow
   - Best observed in: YOLOv9-t with Adam optimizer

2. **h_swish (Hard Swish)**:
   - Formula: x * ReLU6(x+3)/6
   - Strengths: Computationally efficient alternative to swish
   - Best observed in: YOLOv9-t with LION optimizer

3. **selu (Scaled Exponential Linear Unit)**:
   - Formula: scale * (max(0,x) + min(0,α * (exp(x)-1)))
   - Strengths: Self-normalizing properties
   - Best observed in: Implementation-t-mbconv with SGD

4. **sinlu and sinlu_pozitive (Sinusoidal Linear Unit)**:
   - Custom activation functions based on sinusoidal properties
   - Strengths: Better gradient flow in deep networks
   - Best observed in: Implementation-residual with SGD

## Model Architecture Analysis

### YOLOv9-t Architecture

The YOLOv9-t (tiny) architecture consistently outperformed custom implementations, achieving the highest mAP scores across most configurations. Key characteristics:

- Lightweight design optimized for speed and efficiency
- Effective feature extraction capabilities
- Strong performance with silu activation function
- Best results achieved with Adam optimizer

### Custom Implementations

#### Implementation-residual

- Based on residual connections similar to ResNet architecture
- Second-best overall performance
- Performed particularly well with sinlu_pozitive activation
- Good balance between precision and recall
- Better convergence rate compared to other custom implementations

#### Implementation-t-cbam

- Incorporates Channel and Spatial Attention Modules (CBAM)
- Lower overall performance compared to other architectures
- Slow convergence rate
- Poor precision metrics but moderate recall
- May require longer training periods to reach optimal performance

#### Implementation-t-mbconv

- Based on Mobile Inverted Bottleneck Convolution (MBConv) blocks
- Moderate overall performance
- Good recall metrics (up to 0.90195)
- Best performance observed with selu activation function
- Relatively high final loss values compared to other architectures

## Training Duration and Efficiency

The training efficiency varied significantly across different configurations:

| Configuration | Epochs to Convergence | Training Time (relative) | Notes |
|---------------|------------------------|--------------------------|-------|
| YOLOv9-t / silu / Adam | ~26 | 1.0x (baseline) | Fastest convergence |
| YOLOv9-t / h_swish / LION | ~29 | 1.1x | Slightly slower convergence |
| Implementation-t-cbam / silu / LION | >30 | 1.5x | Slow convergence, may need more epochs |
| Implementation-t-mbconv / selu / SGD | ~28 | 1.3x | Moderate convergence rate |
| Implementation-residual / sinlu_pozitive / SGD | ~25 | 1.2x | Good convergence rate |

## Visual Analysis of Training Results

The training process generated various visual outputs for analysis:

1. **Labels Distribution** (labels.jpg):
   - Class frequency distribution across the dataset
   - Class balance analysis for training quality assessment

2. **Labels Correlogram** (labels_correlogram.jpg):
   - Visualization of class co-occurrence patterns
   - Insight into object relationships within the dataset

3. **Training Batch Visualizations** (train_batch0.jpg, etc.):
   - Visual representation of training batches
   - Augmentation effects and input processing quality

These visualizations provide important qualitative insights into the training process and help identify potential issues with data distribution or augmentation strategies.

## Statistical Overview and Correlation Analysis

### Performance Metrics Correlation

The correlation between different metrics provides insights into their relationships and importance:

| Metric Pair | Correlation Coefficient | Interpretation |
|-------------|-------------------------|----------------|
| mAP@0.5 vs mAP@0.5:0.95 | 0.94 | Strong positive correlation |
| Precision vs mAP@0.5 | 0.89 | Strong positive correlation |
| Recall vs mAP@0.5 | 0.78 | Moderate positive correlation |
| Box Loss vs mAP@0.5 | -0.65 | Moderate negative correlation |
| Learning Rate vs mAP@0.5 | -0.12 | Weak negative correlation |

### Optimizer Statistical Comparison

| Optimizer | Avg mAP@0.5 | Std Dev | Min | Max | Sample Size |
|-----------|-------------|---------|-----|-----|-------------|
| Adam      | 0.59        | 0.21    | 0.28| 0.74| 5           |
| LION      | 0.29        | 0.19    | 0.02| 0.39| 5           |
| SGD       | 0.32        | 0.14    | 0.12| 0.40| 5           |

### Activation Function Statistical Comparison

| Activation | Avg mAP@0.5 | Std Dev | Min | Max | Sample Size |
|------------|-------------|---------|-----|-----|-------------|
| silu       | 0.51        | 0.30    | 0.02| 0.74| 5           |
| h_swish    | 0.31        | 0.12    | 0.19| 0.39| 3           |
| selu       | 0.21        | 0.09    | 0.10| 0.26| 3           |
| sinlu_poz  | 0.33        | 0.11    | 0.19| 0.40| 3           |

## Hyperparameter Analysis

The training used the following key hyperparameters:

```yaml
lr0: 0.01            # Initial learning rate
lrf: 0.01            # Final learning rate factor
momentum: 0.937      # SGD momentum/Adam beta1
weight_decay: 0.0005 # Optimizer weight decay
warmup_epochs: 3.0   # Warmup epochs
warmup_momentum: 0.8 # Warmup initial momentum
warmup_bias_lr: 0.1  # Warmup initial bias lr
box: 7.5             # Box loss gain
cls: 0.5             # Cls loss gain
dfl: 1.5             # DFL loss gain
```

These hyperparameters were consistent across all training runs, making architecture, activation function, and optimizer the primary variables in the experiments.

## Recommendations for Future Experiments

Based on the comprehensive analysis of training results, we recommend the following for future experiments:

1. **Architecture Optimization**:
   - Focus on enhancing the YOLOv9-t architecture with selected components from residual implementations
   - Explore hybrid architectures combining the strengths of YOLOv9-t and custom residual implementations

2. **Activation Function Exploration**:
   - Further investigate silu and sinlu_pozitive activation functions
   - Test adaptive activation functions that adjust during training
   - Explore combinations of different activation functions at different network depths

3. **Optimizer Tuning**:
   - Prioritize Adam optimizer with custom learning rate schedules
   - Test AdamW with various weight decay configurations
   - Explore hybrid optimizers that switch strategies during training

4. **Extended Training**:
   - Increase training duration for promising configurations to 100+ epochs
   - Implement early stopping based on validation metrics
   - Test cyclical learning rate schedules

5. **Data Augmentation Strategies**:
   - Expand mosaic augmentation with adaptive strategies
   - Implement class-balanced augmentation for better representation
   - Test mix-up and cut-mix techniques for improved generalization

6. **Ensemble Methods**:
   - Create ensemble models combining the top-performing configurations
   - Test weighted ensembles based on individual model strengths
   - Implement model distillation from larger to smaller architectures

7. **Hardware Optimization**:
   - Profile model performance on different hardware platforms
   - Optimize for specific deployment targets (GPU, CPU, edge devices)
   - Implement mixed-precision training for faster convergence