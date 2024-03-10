import argparse

from torchvision import models
from torchvision.io import read_image
from torchvision.transforms import v2
from torch.nn import Linear
from torch import load, float32, max as tmax, no_grad

parser = argparse.ArgumentParser(
    description="Make defect prediction for a single bean image"
)
parser.add_argument("-w", "--weights-file", help="Path to model weights file")
parser.add_argument("-i", "--input-image", help="Path to input image")
args = vars(parser.parse_args())

transform = v2.Compose(
    [
        v2.Resize(size=(400, 400)),
        v2.ToDtype(float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

weights_path = args["weights_file"]
image_path = args["input_image"]
DEFECT_CLASSES = {
    0: "Burnt",
    1: "Normal (no defect)",
    2: "Bean fragment / broken bean",
    3: "Under roasted",
    4: "Quaker",
    5: "Insect/mold damage",
}
model = models.resnet18()
last_layer_input_features = model.fc.in_features
model.fc = Linear(last_layer_input_features, len(DEFECT_CLASSES))
model.load_state_dict(load(weights_path))
input_image = read_image(image_path).unsqueeze(0)
model.eval()
with no_grad():
    out = model(transform(input_image))

_, pred_class = tmax(out, 1)
print(DEFECT_CLASSES[pred_class.item()])
