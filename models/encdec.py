import torch
import torch.nn as nn
from torchvision.models import alexnet
from typing import List, Tuple

class EncoderDecoder(nn.Module):
    def __init__(self, in_channels: int, bottleneck_dim: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(inplace=False)  # Changed to inplace=False
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(bottleneck_dim, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False)  # Changed to inplace=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AlexNetEncDec(nn.Module):
    def __init__(self, num_classes: int = 102, bottleneck_dims: List[int] = None):
        super().__init__()

        self.alexnet = alexnet(weights='IMAGENET1K_V1')

        # Rebuild features without inplace operations
        new_features = []
        for layer in self.alexnet.features:
            if isinstance(layer, nn.ReLU):
                new_features.append(nn.ReLU(inplace=False))
            else:
                new_features.append(layer)
        self.alexnet.features = nn.Sequential(*new_features)

        # Default bottleneck dimensions for each encoder-decoder
        if bottleneck_dims is None:
            bottleneck_dims = [32, 64, 128, 192, 192]  # 5 dimensions for 5 conv layers

        # Get AlexNet feature layers
        self.feature_layers = list(self.alexnet.features)

        # Create encoder-decoders between conv layers
        self.encoder_decoders = nn.ModuleList()

        # Calculate the correct input features for classifier
        x = torch.randn(1, 3, 64, 64)  # Sample input
        conv_idx = 0

        # Pass through feature layers and encoder-decoders
        for layer in self.feature_layers:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                channel_size = x.shape[1]
                self.encoder_decoders.append(EncoderDecoder(channel_size, bottleneck_dims[conv_idx]))
                x = self.encoder_decoders[conv_idx](x)
                conv_idx += 1

        # Apply avgpool and flatten to get correct classifier input size
        x = self.alexnet.avgpool(x)
        x = torch.flatten(x, 1)
        classifier_input_size = x.shape[1]

        # Rebuild classifier with correct dimensions while keeping same structure
        self.alexnet.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

        # Freeze all parameters first
        for param in self.alexnet.parameters():
            param.requires_grad = False

        # Only unfreeze the final classifier layer
        for param in self.alexnet.classifier[-1].parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep track of conv layer index
        conv_idx = 0

        for layer in self.feature_layers:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                x = self.encoder_decoders[conv_idx](x)
                conv_idx += 1

        x = self.alexnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.alexnet.classifier(x)

        return x

def load_and_prepare_model(config):
    model = AlexNetEncDec()
    return model.to(config['device'])