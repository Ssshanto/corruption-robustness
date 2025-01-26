import argparse
import json
import models.alexnet
from config import get_config
from utils import train, dataloader
import evaluate
import torch
from models import alexnet
import os

# required parameters: epochs, seed, batchsize
# TODO: modify results to include seed, epoch-count

def main(args):
    if not os.path.exists('caltech-101'):
        print("Dataset does not exist: Downloading caltech-101.zip")
        os.system('gdown 1-1qEvCfDf13M_K28tYCuyrfbKlLRBtU4')
        print("Unzipping caltech-101.zip")
        os.system('unzip -q caltech-101.zip')
        print("Deleting caltech-101.zip")
        os.system('rm -rf caltech-101.zip')
    else:
        print("Dataset directory already exists, skipping download")

    config = get_config(args.model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    SEED = args.seed
    dataloader.set_seeds(SEED)
    train.set_seeds(SEED)

    if args.model == 'alexnet':
        model = alexnet.load_and_prepare_model(config)
        print("Loading Model: AlexNet")
    elif args.model == 'encoder_decoder':
        model = encoder_decoder.load_and_prepare_model(config)
    elif args.model == 'mlp_aggregation':
        model = mlp_aggregation.load_and_prepare_model(config)
    else:
        model = recurrent_mlp.load_and_prepare_model(config)

    # Print parameter statistics
    total_params, trainable_params = alexnet.count_parameters(model)
    print("\nModel Parameter Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")

    # Get dataloaders
    dataloaders = dataloader.get_dataloaders(
        root_dir=config['data_dir'],
        mode=config['data_mode'],
        split_file=f"splits/{SEED}.json"
    )

    # Define training parameters
    criterion = config['criterion']
    optimizer = config['optimizer'](filter(lambda p: p.requires_grad, model.parameters()), **config['optimizer_params'])
    scheduler = config['scheduler'](optimizer, **config['scheduler_params'])

    # Train the model
    print(f"\nStarting {args.model} model training...")

    model, history = train.train_model(model, dataloaders, criterion, optimizer, scheduler, args.epochs, device)     

    # Evaluate the model
    print(f"\nEvaluating best {args.model} model...")
    metrics = evaluate.evaluate_model(model, dataloaders['test'])

    # Save training history and evaluation metrics
    results = {
        'history': {k: [float(v) for v in vals] for k, vals in history.items()},
        'evaluation_metrics': metrics
    }

    output_dir = config['output_dir']
    with open(f"{output_dir}/training_results_{SEED}.json", 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {output_dir}/training_results_{SEED}.json")

    # Evaluate the model on the test set
    print(f"\nEvaluating {args.model} model on the test set...")
    test_metrics = evaluate_model(model, dataloaders['test'])

    # Save training history and evaluation metrics
    results = {
        'history': {k: [float(v) for v in vals] for k, vals in history.items()},
        'test_metrics': test_metrics
    }

    output_dir = config['output_dir']
    with open(f"{output_dir}/training_results_{SEED}.json", 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {output_dir}/training_results_{SEED}.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train corruption robustness models')
    parser.add_argument('--model', type=str, default='alexnet', choices=['alexnet', 'encoder_decoder', 'mlp_aggregation', 'recurrent_mlp'],
                        help='Model type to train')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed to set for randomization')
    args = parser.parse_args()
    main(args)
