import torch
import torch.nn as nn
from torchvision import models

def load_and_prepare_model(config):
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    num_classes = config['num_classes']
    model.classifier[6] = nn.Linear(4096, num_classes)
    model.classifier[6].requires_grad = False

    model.classifier[5].requires_grad = False

    return model.to(config['device'])

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

if __name__ == '__main__':
    import torch
    from PIL import Image
    from torchvision import transforms
    
    print("Testing AlexNet-Base")

    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_classes': 102
    }
    
    # Load model
    model = load_and_prepare_model(config)
    model.eval()
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = Image.open('../caltech-101/clean/accordion/image_0001.jpeg')
    input_tensor = transform(image).unsqueeze(0).to(config['device'])
    
    # Get predictions
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
    # Print top prediction
    _, predicted_idx = torch.max(output, 1)
    print(f'Predicted class index: {predicted_idx.item()}')
    print(f'Confidence: {probabilities[predicted_idx].item():.2%}')
