"""
Model Memory Profiler for CIFAR-10 MLP Ablation Study.

This module calculates the theoretical VRAM footprint for PyTorch models,
specifically accounting for parameter weights, gradients, and Adam optimizer states.
"""

import torch.nn as nn
from dataclasses import dataclass


BYTES_IN_MB: int = 1024 ** 2
BYTES_PER_FLOAT32: int = 4

# Multipliers for training memory overhead
# Weights (1x) + Gradients (1x) + Adam Optimizer States (2x: variance & momentum)
TRAINING_MEMORY_MULTIPLIER: int = 4

@dataclass
class ModelMemoryProfile:
    """Data class to store memory profiling metrics for a specific model configuration."""
    hidden_size: int
    total_params: int
    weights_mb: float
    training_mb: float


class CIFAR10MLP(nn.Module):
    """
    Single hidden-layer Multi-Layer Perceptron (MLP) for CIFAR-10 classification.

    Args:
        hidden_size (int): Number of neurons in the hidden layer.
        dropout_prob (float): Probability of an element to be zeroed. Default: 0.0.
    """

    def __init__(self, hidden_size: int, dropout_prob: float = 0.0):
        super().__init__()
        # Input features: 3 channels * 32 * 32 pixels = 3072
        self.fc1 = nn.Linear(3072, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def profile_model_memory(hidden_size: int) -> ModelMemoryProfile:
    """
    Initializes the model and calculates its memory footprint in megabytes.

    Args:
        hidden_size (int): The number of hidden neurons to test.

    Returns:
        ModelMemoryProfile: A dataclass containing the calculated metrics.
    """
    model = CIFAR10MLP(hidden_size=hidden_size)

    # Sum all parameters that require gradient updates
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate bytes
    weights_bytes = total_params * BYTES_PER_FLOAT32
    training_bytes = weights_bytes * TRAINING_MEMORY_MULTIPLIER

    # Convert to Megabytes
    weights_mb = weights_bytes / BYTES_IN_MB
    training_mb = training_bytes / BYTES_IN_MB

    return ModelMemoryProfile(
        hidden_size=hidden_size,
        total_params=total_params,
        weights_mb=weights_mb,
        training_mb=training_mb
    )


def main():
    """
    Runs the memory profiling for a hardcoded list of hidden sizes
    and prints the results simply.
    """
    hidden_sizes = [128, 256, 512, 1024]

    for h_size in hidden_sizes:
        profile = profile_model_memory(hidden_size=h_size)

        # Simple, no-frills print statement
        print(f"Hidden Size: {profile.hidden_size} | "
              f"Total Params: {profile.total_params:,} | "
              f"Weights: {profile.weights_mb:.3f} MB | "
              f"Training Memory: {profile.training_mb:.3f} MB")


if __name__ == "__main__":
    main()