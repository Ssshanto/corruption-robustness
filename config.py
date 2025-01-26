import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

def get_config(model_type):
    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'num_classes': 102,
        'data_dir': 'caltech-101',
        'batch_size': 8192,
        'data_mode': 'standard',
        'criterion': nn.CrossEntropyLoss(),
        'optimizer': optim.Adam,
        'optimizer_params': {
            'lr': 0.001,
        },
        'scheduler': lr_scheduler.StepLR,
        'scheduler_params': {
            'step_size': 7,
            'gamma': 0.1,
        },
        'output_dir': f'results/{model_type}',
    }

    if model_type == 'alexnet':
        config.update({
        })
    elif model_type == 'encoder_decoder':
        config.update({
        })
    elif model_type == 'mlp_aggregation':
        config.update({
        })
    elif model_type == 'recurrent_mlp':
        config.update({
        })

    return config