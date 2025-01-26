import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import numpy as np
from tqdm import tqdm
import torchvision.models as models
import random

def train_model(model, dataloaders, criterion, optimizer, scheduler=None,
                num_epochs=25, device='cpu', save_path, SEED):
    """
    Train the model and perform evaluation

    Args:
        model: PyTorch model
        dataloaders: Dictionary containing 'train' and 'val' dataloaders
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        num_epochs: Number of epochs to train
        device: Device to train on
        save_path: Path to save best model weights

    Returns:
        model: Best performing model
        history: Training history
    """
    model = model.to(device)

    # Initialize history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            # Progress bar for each phase
            pbar = tqdm(dataloaders[phase], desc=f'{phase} epoch {epoch+1}/{num_epochs}')

            # Iterate over data
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    # Debug info
                    if torch.min(labels) < 0 or torch.max(labels) >= outputs.size(1):
                        print(f"Label range error: min={torch.min(labels).item()}, max={torch.max(labels).item()}")
                        print(f"Number of model output classes: {outputs.size(1)}")
                        raise ValueError("Labels out of range for model outputs")

                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)

                # Update progress bar
                pbar.set_postfix({
                    'loss': running_loss / total_samples,
                    'acc': running_corrects.double() / total_samples
                })

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples

            # Record history
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if best validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                if save_path is not None:
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, f"{save_path}/weights_{SEED}.pth")
                    print(f'Saved new best model with validation accuracy: {best_acc:.4f}')

        # Step the scheduler if provided
        if scheduler is not None and phase == 'train':
            scheduler.step()

        print()

    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

def set_seeds(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False