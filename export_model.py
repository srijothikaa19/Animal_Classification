import torch
import torchvision.models as models

# Load the checkpoint
checkpoint = torch.load("models/mobilenetv3_animals10.pth", map_location="cpu")

# Print to see what's inside (optional debug)
print("Keys in checkpoint:", checkpoint.keys())

# Build model
model = models.mobilenet_v3_small(num_classes=5)

# Load only the model weights
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Export to TorchScript
dummy_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("models/mobilenetv3_scripted.pt")

print("Done! Model exported.")
print("Class names found:", checkpoint["class_names"])