from turtle import forward
import torch
from torch import nn

class YOLOVGG16(nn.Module):
    def __init__(self):
        super(YOLOVGG16, self).__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', weights="VGG16_Weights.IMAGENET1K_V1").features
        self.head = nn.Sequential( 
            nn.Flatten(),
            nn.Linear(512 * 13 * 13, 7 * 7 * (2 * 5 + 91)),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)
        predictions = self.head(features)
        return predictions.view(x.shape[0],7,7,2*5+91)