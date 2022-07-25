import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image, ImageReadMode
import argparse
import torch
from torchsummary import summary
from models.vgg16 import YOLOVGG16

# Using Apple Metal to accelerate the training
# The operator 'aten::_slow_conv2d_forward' is not current implemented for the MPS device.
# Check in https://github.com/pytorch/pytorch/issues/77764
# device = "mps" if torch.backends.mps.is_available() else "cpu"


# Default values
device = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THREASHOLD = 0.75


parser = argparse.ArgumentParser(description='Testing arguments')
parser.add_argument("--confidence", "-c", type=float, default=CONFIDENCE_THREASHOLD, help="Confidence Threashold")
parser.add_argument("--images", "-i", type=str, help="Path of the input images")
parser.add_argument("--weights", "-w", type=str, help="Path of the weights of the trained model")

if __name__ == "__main__":
    args = parser.parse_args()
    
    # We should check if the path is a file or a directory
    image = read_image(args.images, ImageReadMode.RGB).to(device)

    model = YOLOVGG16(classes=3)
    model.load_state_dict(args.weights)

    summary(model, (3, 416, 416))
    print("Confidence Threashold: ", args.confidence)
    print("Images Path: ", args.images)
    print("Trained Model Weights: ", args.weights)
    print("----------------------------------------------------------------")

    plt.imshow(torch.permute(image, (1,2,0)))
    plt.show()


