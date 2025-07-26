import torch
from torchvision import transforms
from PIL import Image
import os

def load_image(image_path, image_size=224):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model