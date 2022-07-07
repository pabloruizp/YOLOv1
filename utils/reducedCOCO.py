# Script to generate a reduced version of the COCO dataset
import os
import json

valid_classes = (8,9,16)

conversion_classes = {
    8: 0,
    9: 1,
    16: 2,
}

annotations_folder = "/Users/pabloruizponce/Downloads/annotations"
annotation_file = "instances_val2017.json"
images_folder = "/Users/pabloruizponce/Downloads/val2017"
images = set()
new_annotations = []
new_images = []


f = open(os.path.join(annotations_folder,annotation_file))
data = json.load(f)
f.close()


for i in data["annotations"]:
    if i["category_id"] in valid_classes:
        i["category_id"] = conversion_classes[i["category_id"]]
        new_annotations.append(i)
        images.add(str(i["image_id"]).zfill(12) + ".jpg")

for i in data["images"]:
    if i["file_name"] in images:
        new_images.append(i)

data["annotations"] = new_annotations
data["images"] = new_images

f = open(os.path.join(annotations_folder, "instances_val_reduced.json"), "w")
new_data = json.dumps(data)
f.write(new_data)
f.close()


files = os.listdir(images_folder)

for file in os.listdir(images_folder):
    if file not in images:
        os.remove(os.path.join(images_folder, file))
        print("Removed: ", file)