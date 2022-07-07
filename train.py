import torchvision
import torch
from torch.utils.data import DataLoader
from utils.loss import YOLOLoss
from models.vgg16 import YOLOVGG16
from data.datasets import COCODataset, COCO_collate
from torchsummary import summary
import utils.logger

logger = utils.logger.Logger('YOLOv1')
device = "cuda" if torch.cuda.is_available() else "cpu"

# Using Apple Metal to accelerate the training
# The operator 'aten::_slow_conv2d_forward' is not current implemented for the MPS device.
# Check in https://github.com/pytorch/pytorch/issues/77764
# device = "mps" if torch.backends.mps.is_available() else "cpu"

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device) 
        y = y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y, classes=model.classes)

        logger.log(loss=loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


train_dataset = COCODataset("/Users/pabloruizponce/Downloads/annotations/instances_val_reduced.json", 
                            "/Users/pabloruizponce/Downloads/val2017", 
                            transform=torchvision.transforms.Resize((416,416)))


batch_size = 16
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


model = YOLOVGG16(classes=3).to(device)

# Freeze backbone layers
for param in model.backbone.parameters():
    param.requires_grad = False

summary(model, (3, 416, 416))


loss_fn = YOLOLoss
optimizer = torch.optim.Adam(model.head.parameters())
epochs = 5
logger.config = {
  "learning_rate": 0.001,
  "epochs": epochs,
  "batch_size": batch_size
}

logger.watch(model)

for e in range(epochs):
    print(f"Epoch {e+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
