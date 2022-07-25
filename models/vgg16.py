import torch
from torch import nn

class YOLOVGG16(nn.Module):
    def __init__(self, classes, S=7, B=2):
        super(YOLOVGG16, self).__init__()
        self.classes = classes
        self.S = S
        self.B = B
        #self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', weights="VGG16_Weights.IMAGENET1K_V1").features
        # For compatibility with torch version in Google Colab
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True).features
        self.head = nn.Sequential( 
            nn.Flatten(),
            nn.Linear(512 * 13 * 13, S * S * (B * 5 + classes)),
            nn.Sigmoid()
        )
        # Freeze backbone layers
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)
        predictions = self.head(features)
        return predictions.view(x.shape[0],self.S,self.S,self.B*5+self.classes)