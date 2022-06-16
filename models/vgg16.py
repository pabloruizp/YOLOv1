from turtle import forward
import torch
from torch import nn

class YOLOVGG16(nn.Module):
    def __init__(self):
        super(YOLOVGG16, self).__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True).features
        self.head = nn.Sequential( 
            nn.Flatten(),
            nn.Linear(512 * 13 * 13, 1470)
        )

    def forward(self, x):
        features = self.backbone(x)
        predictions = self.head(features)
        return predictions