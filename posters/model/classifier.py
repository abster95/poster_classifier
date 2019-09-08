import torch
import torch.nn as nn
import torchvision

from torchvision.models import resnet18


class Classifier(nn.Module):
    def __init__(self, backbone=resnet18, imagenet_weights=True, num_classes=20):
        super(Classifier, self).__init__()
        self.backbone = backbone(pretrained=imagenet_weights)
        self.fc = nn.Linear(1000, num_classes)

    def forward(self, img_batch):
        img_batch.cuda()
        x = self.backbone(img_batch)
        return self.fc(x)