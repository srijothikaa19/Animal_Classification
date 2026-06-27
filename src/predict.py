import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "models/mobilenetv3_animals10.pth"

if len(sys.argv) != 2:
    print("Usage: python src\\predict.py path\\to\\image.jpg")
    sys.exit(1)

image_path = sys.argv[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint["class_names"]

model = models.mobilenet_v3_small(weights=None)
num_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(num_features, len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(image)
    probabilities = torch.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

print("Prediction:", class_names[predicted.item()])
print("Confidence:", round(confidence.item() * 100, 2), "%")