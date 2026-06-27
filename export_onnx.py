import torch
import torchvision.models as models

# Load checkpoint
checkpoint = torch.load("models/mobilenetv3_animals10.pth", map_location="cpu")

# Build model
model = models.mobilenet_v3_small(num_classes=5)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "models/mobilenetv3.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}},
    opset_version=11
)

print("Done! ONNX model exported to models/mobilenetv3.onnx")