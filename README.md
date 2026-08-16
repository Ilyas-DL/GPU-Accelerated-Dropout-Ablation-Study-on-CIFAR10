# GPU-Accelerated Dropout Ablation Study on CIFAR-10

This repository contains an end-to-end ablation study observing the regularization effects of Dropout on a Multi-Layer Perceptron (MLP) trained on the CIFAR-10 image dataset. The project heavily emphasizes PyTorch CUDA optimization, specifically by loading the entire dataset into GPU VRAM to eliminate CPU-to-GPU transfer bottlenecks during training.

## Dataset Overview

The project uses the standard CIFAR-10 dataset, which consists of 60,000 color images distributed equally across 10 classes (Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck). 

**Dataset Splits & Sizes:**
*   Training Images: 50,000 (5,000 per class)
*   Validation Images: 10,000
*   Train/Val Split: 83% / 17%

**Structural Dimensions:**
*   Original Image Shape: 32 x 32 pixels
*   Color Channels: 3 (RGB)
*   Flattened Tensor Shape: [3, 32, 32] -> 3072 Input Neurons

**Memory Footprint (Float32 for GPU):**
*   Size per Image: 12.00 KB
*   Training Dataset VRAM: ~585.94 MB
*   Validation Dataset VRAM: ~117.19 MB
*   Total VRAM Required: ~703.12 MB

## Model Architecture

The model is a standard Single Hidden Layer Multi-Layer Perceptron (MLP). The network maps the 3072 input pixels to a specified hidden dimension, applies non-linearity, and outputs the 10 class logits.

**Without Dropout (Reference Base):**
The forward pass is defined mathematically as:
`Output = W_2 * (ReLU(W_1 * X + b_1)) + b_2`

**With Dropout:**
Dropout is applied immediately after the activation function to prevent complex co-adaptations on the training data. The forward pass becomes:
`Output = W_2 * (Dropout(ReLU(W_1 * X + b_1), p)) + b_2`

Where `p` is the probability of an element to be zeroed. During the evaluation phase, the Dropout layer functions as an identity operator.

## Experimental Methodology

The experiment analyzes the relationship between the network capacity (hidden layer size) and the required regularization intensity (dropout probability `p`). 

*   **Network Capacities Tested:** Hidden sizes of 128, 256, 512, and 1024 neurons.
*   **Dropout Rates Tested:** 0.0 (Reference), 0.1, 0.2, 0.3, and 0.4.
*   **Hyperparameters:** 60 epochs per configuration, Adam optimizer, learning rate of 1e-3, and a batch size of 512.

To maximize throughput, the dataset is loaded into GPU VRAM prior to the training loops. Standard PyTorch DataLoaders are bypassed in favor of in-VRAM manual batch slicing and tensor shuffling.

## Results & Visualizations

The following plots detail the Training Cross-Entropy Loss and Validation Accuracy across different dropout rates for each hidden layer dimension. As network capacity increases, the model's tendency to overfit the training data rises, increasing the necessity and effectiveness of higher dropout probabilities.

### Hidden Layer Size: 128
![Dropout Ablation Hidden 128](plots/dropout_ablation_hidden_128.png)

### Hidden Layer Size: 256
![Dropout Ablation Hidden 256](plots/dropout_ablation_hidden_256.png)

### Hidden Layer Size: 512
![Dropout Ablation Hidden 512](plots/dropout_ablation_hidden_512.png)

### Hidden Layer Size: 1024
![Dropout Ablation Hidden 1024](plots/dropout_ablation_hidden_1024.png)

### Global VRAM Usage Over Time
To ensure the CUDA memory optimizations remain stable across multiple initializations, global VRAM allocation was tracked throughout the execution of the entire experiment. Red markers indicate the transition between different hidden layer size runs.

![Global VRAM Usage](plots/global_vram_usage_over_time.png)

## Running the Code

The complete study is encapsulated within a single script. It requires a CUDA-enabled machine to run effectively.

```bash
# Clone the repository
git clone https://github.com/Ilyas-DL/GPU-Accelerated-Dropout-Ablation-Study-on-CIFAR10.git
cd GPU-Accelerated-Dropout-Ablation-Study-on-CIFAR10

# Run the experimental script
python Pytorch-CIFAR10-experiments.py
