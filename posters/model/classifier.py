import torch
import torch.nn as nn
import torchvision

from torchvision.models import resnet101


class Classifier(nn.Module):
    def __init__(self, backbone=resnet101, imagenet_weights=True, num_classes=20):
        super(Classifier, self).__init__()
        self.backbone = backbone(pretrained=imagenet_weights)
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, img_batch):
        x = self.backbone(img_batch)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.fc4(x)

