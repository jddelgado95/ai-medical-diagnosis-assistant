import torch
from src.model import build_model
from src.dataloader import get_dataloaders
from src.train import train_model
from src.utils import save_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = "data"  # points to your data/ folder
train_loader, val_loader, class_names = get_dataloaders(data_dir, batch_size=32)
num_classes = len(class_names)

model = build_model(num_classes=num_classes, pretrained=True)

train_model(model, train_loader, val_loader, device, epochs=10, lr=0.0001)

save_checkpoint(model, "weights/model.pth")
