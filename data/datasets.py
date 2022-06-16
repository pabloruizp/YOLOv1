#COCO Dataset
import os
import json
from tkinter.ttk import LabeledScale
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from torchvision.io import read_image

# Custom collate function for correctly returning labels
def COCO_collate(data):
    images = [i[0] for i in data]
    labels = [i[1] for i in data]
    return [images, labels]

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
        label = label.to_dict('records')
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label

