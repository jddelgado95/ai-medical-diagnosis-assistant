import torch.nn as nn
import torchvision.models as models

def build_model(num_classes, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model