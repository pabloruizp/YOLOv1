import torchvision
import torch
import os
from torch.utils.data import DataLoader
from utils.loss import YOLOLoss
from models.vgg16 import YOLOVGG16
from data.datasets import COCODataset
from torchsummary import summary
import utils.logger
from tqdm import tqdm

# Using Apple Metal to accelerate the training
# The operator 'aten::_slow_conv2d_forward' is not current implemented for the MPS device.
# Check in https://github.com/pytorch/pytorch/issues/77764
# device = "mps" if torch.backends.mps.is_available() else "cpu"

logger = utils.logger.Logger('YOLOv1')
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
epochs = 10
lr = 1e-4
weight_decay = 0.0005
images_path = "/Users/pabloruizponce/Downloads/val2017"
annotations_path = "/Users/pabloruizponce/Downloads/annotations/instances_val_reduced.json"




train_dataset = COCODataset(annotations_path, 
                            images_path, 
                            transform=torchvision.transforms.Compose(
                                [torchvision.transforms.Resize((416,416)),
                                 torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])
                                ]))

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


model = YOLOVGG16(classes=3).to(device)
loss_fn = YOLOLoss
optimizer = torch.optim.SGD(model.head.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)



# 1e-3 = 1e-4 * X^50   -> X = 10 ** (1./50.)
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,gamma=(10 ** (1./50.)),step_size=1)
scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,gamma=0.1,milestones=[75,105,135])
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer, schedulers=[scheduler1,scheduler2], milestones=[50])


logger.config = {
  "learning_rate": lr,
  "weight_decay": weight_decay,
  "epochs": epochs,
  "batch_size": batch_size
}

logger.watch(model)


def train(dataloader, model, loss_fn, optimizer, epoch_n=0):
    size = len(dataloader)
    model.train()

    with tqdm(dataloader, unit="batch", bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}') as tepoch:
        
        e_loss = torch.tensor([0,0,0,0,0,0], dtype=torch.float64).to(device)
        
        for batch, (X, y) in enumerate(tepoch):

            tepoch.set_description(f"Epoch {epoch_n}")

            X = X.to(device) 
            y = y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y, device, classes=model.classes)
            e_loss += loss / size
            

            # Backpropagation
            optimizer.zero_grad()
            loss[0].backward()
            optimizer.step()

            tepoch.set_postfix(loss=e_loss[0].item(),lr=scheduler.get_last_lr()[0])

        logger.log(loss=e_loss[0],
                   xy_loss=e_loss[1], 
                   wh_loss=e_loss[2], 
                   obj_loss=e_loss[3], 
                   noobj_loss=e_loss[4], 
                   class_loss=e_loss[5],
                   lr=scheduler.get_last_lr()[0])




if __name__ == "__main__":
    summary(model, (3, 416, 416))

    if not os.path.exists("./YOLOoutputs"):
        os.makedirs("./YOLOoutputs")

    os.makedirs("./YOLOoutputs/" + logger.name)

    for e in range(epochs):
        train(train_dataloader, model, loss_fn, optimizer, epoch_n=e+1)
        torch.save(model.state_dict(), "./YOLOoutputs/" + logger.name + "/epoch-" + str(e+1) + ".pth")
        scheduler.step()
