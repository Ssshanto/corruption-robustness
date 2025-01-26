import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.dataloader import get_dataloaders

def evaluate_model(model, test_loader, device):
    """
    Evaluate model performance with detailed corruption and severity level analysis

    Args:
        model: PyTorch model
        test_loader: DataLoader containing test data
        device: Device to run evaluation on

    Returns:
        dict: Dictionary containing various performance metrics
    """
    model.eval()
    model = model.to(device)

    # Initialize counters for different types of aggregation
    results = {
        'per_corruption': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'per_level': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'per_corruption_level': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'clean': {'correct': 0, 'total': 0},
        'corrupted': {'correct': 0, 'total': 0},
        'overall': {'correct': 0, 'total': 0}
    }

    # Create corruption type list
    corruption_types = set()
    corruption_levels = set()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Get batch data and metadata
            images, labels = batch[0], batch[-1]  # Works for both standard and contrastive
            images = images.to(device)
            labels = labels.to(device)

            # Get predictions
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Get metadata for the batch
            batch_size = images.size(0)
            for i in range(batch_size):
                # Get sample metadata
                sample = test_loader.dataset.samples[test_loader.dataset.indices[i] if hasattr(test_loader.dataset, 'indices') else i]
                is_correct = (predicted[i] == labels[i]).item()

                # Update overall statistics
                results['overall']['total'] += 1
                results['overall']['correct'] += is_correct

                if sample['is_clean']:
                    # Update clean statistics
                    results['clean']['total'] += 1
                    results['clean']['correct'] += is_correct
                else:
                    # Update corrupted statistics
                    results['corrupted']['total'] += 1
                    results['corrupted']['correct'] += is_correct

                    # Get corruption details
                    corruption = sample['corruption_type']
                    level = sample['corruption_level']
                    corruption_types.add(corruption)
                    corruption_levels.add(level)

                    # Update per-corruption statistics
                    results['per_corruption'][corruption]['total'] += 1
                    results['per_corruption'][corruption]['correct'] += is_correct

                    # Update per-level statistics
                    results['per_level'][level]['total'] += 1
                    results['per_level'][level]['correct'] += is_correct

                    # Update per-corruption-level statistics
                    key = f"{corruption}_level{level}"
                    results['per_corruption_level'][key]['total'] += 1
                    results['per_corruption_level'][key]['correct'] += is_correct

    # Calculate accuracies
    metrics = {
        'clean_accuracy': 100 * results['clean']['correct'] / results['clean']['total'],
        'corrupted_accuracy': 100 * results['corrupted']['correct'] / results['corrupted']['total'],
        'overall_accuracy': 100 * results['overall']['correct'] / results['overall']['total'],

        'per_corruption_accuracy': {
            corruption: 100 * results['per_corruption'][corruption]['correct'] /
                       results['per_corruption'][corruption]['total']
            for corruption in corruption_types
        },

        'per_level_accuracy': {
            level: 100 * results['per_level'][level]['correct'] /
                   results['per_level'][level]['total']
            for level in corruption_levels
        },

        'per_corruption_level_accuracy': {
            key: 100 * results['per_corruption_level'][key]['correct'] /
                 results['per_corruption_level'][key]['total']
            for key in results['per_corruption_level'].keys()
        }
    }

    return metrics

def plot_corruption_analysis(metrics, output_dir):
    """
    Create visualizations for model performance across corruptions and severity levels

    Args:
        metrics: Dictionary containing evaluation metrics
        output_dir: Directory to save the heatmap image
    """
    # 1. Plot per-corruption accuracies
    plt.figure(figsize=(15, 6))
    corruptions = list(metrics['per_corruption_accuracy'].keys())
    accuracies = list(metrics['per_corruption_accuracy'].values())

    plt.subplot(1, 2, 1)
    plt.bar(corruptions, accuracies)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy per Corruption Type')

    # 2. Plot per-level accuracies
    plt.subplot(1, 2, 2)
    levels = sorted(list(metrics['per_level_accuracy'].keys()))
    level_accuracies = [metrics['per_level_accuracy'][level] for level in levels]

    plt.plot(levels, level_accuracies, marker='o')
    plt.xlabel('Corruption Level')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Corruption Level')

    plt.tight_layout()
    plt.show()

    # 3. Create heatmap for corruption-level combinations
    corruption_level_data = []
    for key, accuracy in metrics['per_corruption_level_accuracy'].items():
        corruption, level = key.split('_level')
        corruption_level_data.append({
            'Corruption': corruption,
            'Level': int(level),
            'Accuracy': accuracy
        })

    df = pd.DataFrame(corruption_level_data)
    pivot_table = df.pivot(index='Corruption', columns='Level', values='Accuracy')

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlOrRd')
    plt.title('Accuracy Heatmap: Corruption Types vs Levels')
    plt.savefig(output_dir)
    plt.close()

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Clean Accuracy: {metrics['clean_accuracy']:.2f}%")
    print(f"Corrupted Accuracy: {metrics['corrupted_accuracy']:.2f}%")
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.2f}%")

    # Print best and worst cases
    print("\nBest performing corruption:",
          max(metrics['per_corruption_accuracy'].items(), key=lambda x: x[1])[0])
    print("Worst performing corruption:",
          min(metrics['per_corruption_accuracy'].items(), key=lambda x: x[1])[0])

def test_model(model, test_loader, output_dir, device):
    """
    Main function to test model and visualize results

    Args:
        model: PyTorch model
        test_loader: DataLoader containing test data
        device: Device to run evaluation on
    """
    print("Starting model evaluation...")
    metrics = evaluate_model(model, test_loader, device)
    print("\nGenerating performance visualizations...")
    plot_corruption_analysis(metrics, output_dir)
    return metrics

if __name__ == "__main__":
    import torchvision.models as models
    from torch import nn

    # Load AlexNet with pretrained weights
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.classifier[6] = nn.Linear(4096, 102)

    # Create test dataloader
    test_loader = get_dataloaders(
        root_dir="caltech-101",
        mode='standard',
        batch_size=32,
        split_file="splits.json"
    )['test']

    print("Starting evaluation of AlexNet on Caltech-101...")
    print(f"Device: {device}")
    metrics = test_model(model, test_loader, device)

    # Save metrics to file
    results_file = "alexnet_results.json"
    with open(results_file, 'w') as f:
        # Convert numpy floats to Python floats for JSON serialization
        serializable_metrics = {
            k: (
                {k2: float(v2) for k2, v2 in v.items()}
                if isinstance(v, dict) else float(v)
            )
            for k, v in metrics.items()
        }
        json.dump(serializable_metrics, f, indent=4)
    print(f"\nResults saved to {results_file}")