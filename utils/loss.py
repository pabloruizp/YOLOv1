from cmath import exp
from multiprocessing import reduction
from traceback import print_tb
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

def whichCell(x, y, S=7):
    return int(y / float(1/S)), int(x / float(1/S))

# This is calculated assuming always a B=2
def YOLOLoss(x, y, device, classes, S=7, B=2, lambda_coord=5, lamda_noobj=.5):

    loss = torch.tensor([0,0,0,0,0,0], dtype=torch.float64).to(device)
    
    expected = torch.Tensor(x.shape).to(device)

    # For each batch element
    for bn, image in enumerate(y):
        
        obj_mask = torch.zeros((S,S), dtype=torch.bool).to(device)
        bbox_mask = torch.zeros((S,S,B*5+classes), dtype=torch.bool).to(device)      
        
        for on, object in enumerate(image): 

            if object[0] < 0:
                continue

            # Get the cell of the object
            Srow, Scol = whichCell(object[0], object[1], S=S)
            obj_mask[Srow][Scol] = 1
            # Check which prediction has a bigger IoU with the target
            iou1 = IoU(object[:4], x[bn][Srow][Scol][:4])
            iou2 = IoU(object[:4], x[bn][Srow][Scol][5:9])

            if iou1 >= iou2:
                    expected[bn][Srow][Scol][:4] = object[:4]
                    expected[bn][Srow][Scol][4] = iou1
                    expected[bn][Srow][Scol][10:] = 0
                    expected[bn][Srow][Scol][10 + object[4].long()] = 1
                    bbox_mask[Srow][Scol][:5] = True
            else:
                    expected[bn][Srow][Scol][5:9] = object[:4]
                    expected[bn][Srow][Scol][9] = iou2
                    expected[bn][Srow][Scol][10:] = 0
                    expected[bn][Srow][Scol][10 + object[4].long()] = 1
                    bbox_mask[Srow][Scol][5:10] = True

        
        # This is so confusing!!
        # We remove all the useless bounding boxes using a mask
        # The resulted tensor is reshaped (#objects,[x,y,w,h,c])
        n_predictions = int(expected[bn][bbox_mask].shape[0] / 5)
        
        xy_loss = F.mse_loss(expected[bn][bbox_mask].view(n_predictions,5)[:,:2], 
                             x[bn][bbox_mask.clone()].view(n_predictions,5)[:,:2],
                             reduction="sum")

        wh_loss = F.mse_loss(torch.sqrt(expected[bn][bbox_mask].view(n_predictions,5)[:,2:4]), 
                             torch.sqrt(x[bn][bbox_mask.clone()].view(n_predictions,5)[:,2:4]),
                             reduction="sum")

        obj_loss = F.mse_loss(expected[bn][bbox_mask].view(n_predictions,5)[:,4], 
                              x[bn][bbox_mask.clone()].view(n_predictions,5)[:,4],
                              reduction="sum")

        inverse_bbox_mask = bbox_mask
        inverse_bbox_mask[:,:,:10] = ~inverse_bbox_mask[:,:,:10]
        noobj_loss = F.mse_loss(torch.zeros(S*S*B-n_predictions).to(device), 
                                x[bn][inverse_bbox_mask].view(S*S*B-n_predictions,5)[:,4],
                                reduction="sum")
    
        class_loss = F.mse_loss(expected[bn][obj_mask][:,10:], 
                                x[bn][obj_mask][:,10:],
                                reduction="sum")


        # Total loss / batch size
        loss[0] += (lambda_coord * (xy_loss + wh_loss) + obj_loss + lamda_noobj * noobj_loss + class_loss) / x.shape[0]
        loss[1] += xy_loss / x.shape[0]
        loss[2] += wh_loss / x.shape[0]
        loss[3] += obj_loss / x.shape[0]
        loss[4] += noobj_loss / x.shape[0]
        loss[5] += class_loss / x.shape[0]
    
    return loss

