import torchvision
import torchvision.transforms as transforms
from collections import Counter
import warnings

# ---------------------------------------------------------
# 1. FIX DEPRECATION WARNINGS
# Suppress the specific NumPy/Pickle align=0 warning from torchvision
# ---------------------------------------------------------
warnings.filterwarnings("ignore", message=".*align should be passed.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)


def print_header(title: str):
    print(f"\n{'-' * 50}")
    print(f" {title}")
    print(f"{'-' * 50}")


def main():
    print("Loading CIFAR-10 Dataset for Analysis...")

    # 1. Load Raw Data (to inspect original pixel values and classes)
    raw_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    raw_val = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

    # 2. Load Transformed Data (to inspect what the neural network actually sees)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    tensor_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)

    # --- METRICS ANALYSIS ---

    print_header("1. DATASET SPLITS & SIZES")
    num_train = len(raw_train)
    num_val = len(raw_val)
    total_images = num_train + num_val
    print(f"Training Images   : {num_train:,}")
    print(f"Validation Images : {num_val:,}")
    print(f"Total Images      : {total_images:,}")
    print(f"Train/Val Split   : {num_train / total_images * 100:.0f}% / {num_val / total_images * 100:.0f}%")

    print_header("2. STRUCTURAL DIMENSIONS (THE MLP INPUT)")
    sample_tensor, _ = tensor_train[0]
    c, h, w = sample_tensor.shape
    flattened_size = sample_tensor.numel()
    print(f"Original Image Shape : {h} x {w} pixels")
    print(f"Color Channels       : {c} (RGB)")
    print(f"Tensor Shape         : [{c}, {h}, {w}]")
    print(f"Neurons in Input Layer   : {flattened_size}")

    print_header("3. MEMORY FOOTPRINT (FLOAT32 FOR GPU)")
    # Memory calculation: Number of images * flattened size * 4 bytes (Float32)
    bytes_per_image = flattened_size * 4
    train_mb = (num_train * bytes_per_image) / (1024 ** 2)
    val_mb = (num_val * bytes_per_image) / (1024 ** 2)
    print(f"Size of 1 Image (fp32) : {bytes_per_image / 1024:.2f} KB")
    print(f"Train Dataset RAM/VRAM : {train_mb:.2f} MB")
    print(f"Val Dataset RAM/VRAM   : {val_mb:.2f} MB")
    print(f"Total GPU VRAM Needed  : {train_mb + val_mb:.2f} MB")

    print_header("4. CLASS DISTRIBUTION")
    classes = raw_train.classes
    target_counts = Counter(raw_train.targets)
    print(f"Total Number of Classes : {len(classes)}")
    print("Class Breakdown (Training Set):")

    # Check if dataset is perfectly balanced
    is_balanced = len(set(target_counts.values())) == 1

    for class_idx, count in sorted(target_counts.items()):
        class_name = classes[class_idx].capitalize()
        print(f" - {class_name:<12} : {count:,} images")

if __name__ == "__main__":
    main()