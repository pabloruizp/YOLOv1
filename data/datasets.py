#COCO Dataset
import os
import json
import pandas as pd
from matplotlib.pyplot import axis
from torch.utils.data import Dataset
from torchvision.io import read_image

# Custom collate function for correctly returning labels
def COCO_collate(data):
    images = [i[0] for i in data]
    labels = [i[1] for i in data]
    return [images, labels]

def convert2YOLOFormat(x, **kwargs):
    bbox = x
    bbox[0] = (bbox[0] + (bbox[2] / 2.)) / kwargs['width']
    bbox[1] = (bbox[1] + (bbox[3] / 2.)) / kwargs['height']
    bbox[2] /= kwargs['width']
    bbox[3] /= kwargs['height']
    return bbox

class COCODataset(Dataset):
    def __init__(self, annotations_file, image_dir, transform=None, target_transform=None):

        f = open(annotations_file)
        data = json.load(f)
        f.close()

        self.img_labels = pd.DataFrame.from_dict(data["images"])
        self.ann_labels = pd.DataFrame.from_dict(data["annotations"])
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx): 
        img_path = os.path.join(self.image_dir, self.img_labels.iloc[idx]["file_name"])
        image = read_image(img_path)

        label = self.ann_labels[self.ann_labels["image_id"] == self.img_labels.iloc[idx].id]
        label = label[["category_id", "bbox"]]
        label['bbox'] = label['bbox'].apply(convert2YOLOFormat, width=self.img_labels.iloc[idx].width, height=self.img_labels.iloc[idx].height)
        label = label.to_dict('records')
        
        for _ in range(7*7*2 - len(label)):
            label.append({})
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label

    




