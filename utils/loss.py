from multiprocessing import reduction
import torch
import torch.nn.functional as F


def convertBBox(a): 
    x1 = a[0] - (a[2] / 2.)
    x2 = a[0] + (a[2] / 2.)
    y1 = a[1] - (a[3] / 2.)
    y2 = a[1] + (a[3] / 2.)
    return torch.tensor([x1, x2, y1, y2])

def intersection(a,b):
    a = convertBBox(a)
    b = convertBBox(b)

    x1 = max(a[0], b[0])
    x2 = min(a[1], b[1])
    y1 = max(a[2], b[2])
    y2 = min(a[3], b[3])

    return (x2 - x1) * (y2 - y1)

def areaBBox(a): 
    return a[2] * a[3]

def IoU(a,b):
    i = intersection(a,b)
    u = areaBBox(a) + areaBBox(b) - i
    return float(i / u)

def YOLOLoss(x, y, lambda_coord=5, lamda_noobj=.5):
    xy_loss = F.mse(x[:,:,:2], y[:,:,:2],reduction="sum")
    wh_loss = F.mse(x[:,:,2:4], y[:,:,2:4],reduction="sum")
    obj_loss = F.mse(x[:,:,4], y[:,:,4],reduction="sum")
    noobj_loss = F.mse(x[:,:,4], y[:,:,4],reduction="sum")
    class_loss = F.mse(x[:,:,5:], y[:,:,5:],reduction="sum")

    loss = lambda_coord * (xy_loss + wh_loss) + obj_loss + lamda_noobj * noobj_loss + class_loss
    return loss
