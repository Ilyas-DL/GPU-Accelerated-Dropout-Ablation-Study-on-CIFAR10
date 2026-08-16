import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import warnings
import psutil
import os
from typing import Dict, Tuple, List

# ---------------------------------------------------------
# 1. FIX DEPRECATION WARNINGS
# Suppress the specific NumPy/Pickle align=0 warning from torchvision
# ---------------------------------------------------------
warnings.filterwarnings("ignore", message=".*align should be passed.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configuration & Hyperparameters
EPOCHS = 60
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
HIDDEN_SIZES = [128, 256, 512,1024]
DROPOUT_RATES = [0.0, 0.1, 0.2, 0.3, 0.4]

# Consistent styling for plots
COLORS = {
    0.0: '#000000',  # Black for Reference
    0.1: '#1f77b4',  # Blue
    0.2: '#2ca02c',  # Green
    0.3: '#ff7f0e',  # Orange
    0.4: '#d62728'  # Red
}

LABELS = {
    0.0: "Reference (p=0.0)",
    0.1: "Dropout (p=0.1)",
    0.2: "Dropout (p=0.2)",
    0.3: "Dropout (p=0.3)",
    0.4: "Dropout (p=0.4)"
}


class SingleLayerMLP(nn.Module):
    """
    Single Hidden Layer MLP: W_2(Dropout(ReLU(W_1X + b1))) + b2
    """

    def __init__(self, input_size: int, hidden_size: int, num_classes: int, dropout_p: float):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_cifar10_to_gpu(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loads the entirety of CIFAR-10 directly into VRAM for maximum throughput.
    """
    print("Loading CIFAR-10 dataset into GPU memory...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Convert entire datasets to tensors
    X_train = torch.stack([x for x, _ in train_dataset]).view(len(train_dataset), -1).to(device)
    Y_train = torch.tensor([y for _, y in train_dataset]).to(device)

    X_val = torch.stack([x for x, _ in val_dataset]).view(len(val_dataset), -1).to(device)
    Y_val = torch.tensor([y for _, y in val_dataset]).to(device)

    return X_train, Y_train, X_val, Y_val


def train_model(
        model: nn.Module,
        X_train: torch.Tensor, Y_train: torch.Tensor,
        X_val: torch.Tensor, Y_val: torch.Tensor,
        epochs: int, batch_size: int,
        start_time: float, vram_times: List[float], vram_gbs: List[float]
) -> Tuple[list, list]:
    """
    Trains the model using in-VRAM manual batch slicing.
    Simultaneously logs VRAM usage at the end of each epoch.
    """
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    num_train = X_train.shape[0]

    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()

        # In-VRAM shuffling
        indices = torch.randperm(num_train, device=X_train.device)
        X_train_shuffled = X_train[indices]
        Y_train_shuffled = Y_train[indices]

        epoch_loss = 0.0
        batches = 0

        for i in range(0, num_train, batch_size):
            x_batch = X_train_shuffled[i:i + batch_size]
            y_batch = Y_train_shuffled[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        train_losses.append(epoch_loss / batches)

        # Validation Phase
        model.eval()
        with torch.no_grad():
            outputs = model(X_val)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == Y_val).sum().item()
            val_acc = (correct / Y_val.size(0)) * 100.0
            val_accuracies.append(val_acc)

        # Log VRAM and time after every epoch
        vram_times.append(time.time() - start_time)
        vram_gbs.append(torch.cuda.memory_allocated() / (1024 ** 3))


    return train_losses, val_accuracies


def plot_and_save_results(results: Dict[float, dict], hidden_size: int):
    """
    Generates and saves the left (Train Error) and right (Val Accuracy) subplots.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Dropout Ablation Study - Hidden Layer Size: {hidden_size}", fontsize=16)

    epochs_range = range(1, EPOCHS + 1)

    for p in DROPOUT_RATES:
        loss = results[p]['loss']
        acc = results[p]['acc']

        ax1.plot(epochs_range, loss, label=LABELS[p], color=COLORS[p], linewidth=2)
        ax2.plot(epochs_range, acc, label=LABELS[p], color=COLORS[p], linewidth=2)

    ax1.set_title("Training Error (Loss)")
    ax1.set_xlabel("Epochs (Iterations)")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epochs (Iterations)")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    filename = f"dropout_ablation_hidden_{hidden_size}.png"
    plt.savefig(filename, dpi=300)
    print(f"[*] Saved plot: {os.path.abspath(filename)}")
    plt.close()


def plot_vram_usage(vram_times: List[float], vram_gbs: List[float], markers: List[Tuple[float, str]]):
    """
    Plots global VRAM usage (in GB) vs Time (in seconds).
    Adds dotted red lines marking the boundaries of different experiments.
    """
    plt.figure(figsize=(12, 6))

    # Plot the VRAM curve
    plt.plot(vram_times, vram_gbs, color='blue', linewidth=2.5, label="Allocated VRAM (GB)")

    # 1. Dynamically determine the maximum VRAM to scale limits
    max_vram = max(vram_gbs)
    plt.ylim(0, max_vram * 1.1)  # Adds 10% headroom above the highest peak

    # Add vertical dotted red lines for milestones
    for i, (t, label) in enumerate(markers):
        # Only add the label to the legend once to keep it clean
        legend_label = "Experiment Boundary" if i == 0 else ""
        plt.axvline(x=t, color='red', linestyle=':', linewidth=2, label=legend_label)

        # 2. Place text dynamically near the top of the new scaled graph
        plt.text(t + (max(vram_times) * 0.01), max_vram * 1.05, label, rotation=90, color='red',
                 verticalalignment='top', fontsize=10, fontweight='bold')

    # Formatting
    plt.title(f"GPU VRAM Usage Evolution Over Time, Batch size = {BATCH_SIZE}", fontsize=15)
    plt.xlabel("Time Elapsed (Seconds)", fontsize=12)
    plt.ylabel("VRAM Usage (GB)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc="upper left", fontsize=11)

    plt.tight_layout()
    filename = "global_vram_usage_over_time.png"
    plt.savefig(filename, dpi=300)
    print(f"[*] Saved VRAM plot: {os.path.abspath(filename)}")
    plt.close()



def main():
    # Set Seed for Determinism
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not torch.cuda.is_available():
        raise SystemError("CUDA is not available! Please run this on your RTX GPU.")

    device = torch.device("cuda")
    print(f"[*] Using Device: {torch.cuda.get_device_name(device)}")

    torch.cuda.reset_peak_memory_stats()

    # VRAM Tracking variables
    global_start_time = time.time()
    vram_times = [0.0]
    vram_gbs = [torch.cuda.memory_allocated() / (1024 ** 3)]
    experiment_markers = []

    # 2. Data Loading (100% on GPU)
    X_train, Y_train, X_val, Y_val = load_cifar10_to_gpu(device)
    input_size = X_train.shape[1]
    num_classes = 10

    # Log VRAM right after loading the dataset
    vram_times.append(time.time() - global_start_time)
    vram_gbs.append(torch.cuda.memory_allocated() / (1024 ** 3))

    # 3. Experiment Runner
    for hidden_size in HIDDEN_SIZES:
        print(f"\n--- Starting Experiments for Hidden Size: {hidden_size} ---")

        # Mark the beginning of a new hidden size experiment
        current_t = time.time() - global_start_time
        experiment_markers.append((current_t, f"Start H={hidden_size}"))

        results = {}

        for p in DROPOUT_RATES:
            print(f"Training Model -> Dropout p={p} ...", end=" ", flush=True)

            model = SingleLayerMLP(input_size, hidden_size, num_classes, p).to(device)

            train_losses, val_accuracies = train_model(
                model, X_train, Y_train, X_val, Y_val,
                epochs=EPOCHS, batch_size=BATCH_SIZE,
                start_time=global_start_time, vram_times=vram_times, vram_gbs=vram_gbs
            )

            results[p] = {
                'loss': train_losses,
                'acc': val_accuracies
            }
            print("Done!")

        plot_and_save_results(results, hidden_size)

    # Mark the end of the entire experiment cycle
    end_t = time.time() - global_start_time
    experiment_markers.append((end_t, "End of Experiments"))

    # 4. Final Global VRAM Plot
    plot_vram_usage(vram_times, vram_gbs, experiment_markers)

    # 5. Metrics Reporting
    total_time = time.time() - global_start_time
    max_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert bytes to MB

    print("\n=============================================")
    print("           EXPERIMENT COMPLETED              ")
    print("=============================================")
    print(f"Total Execution Time : {total_time:.2f} seconds")
    print(f"Peak VRAM Used       : {max_vram:.2f} MB")
    print("=============================================")


if __name__ == "__main__":
    main()
