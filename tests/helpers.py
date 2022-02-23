import sys

import torch
from torch import nn
import albumentations as A

simple_augmentation = A.Compose([
    A.Resize(40, 40)
])


class Demo_model(nn.Module):
    def __init__(self):
        super(Demo_model, self).__init__()
        self.model = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.MaxPool2d(2, 2),
            nn.MaxPool2d(2, 2),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(2*2*3, 3)
        )

    def forward(self, images):
        logits = self.model(images)
        return logits