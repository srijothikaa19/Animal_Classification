#  Animals10 Classifier

> Image classification app using **MobileNetV3** trained on 5 animal classes, with both Python and **C++ inference** support via LibTorch and ONNX export.

---

##  Supported Classes

| Class | Italian Label |
|-------|--------------|
|  Butterfly | farfalla |
|  Cat | gatto |
|  Dog | cane |
|  Elephant | elefante |
|  Horse | cavallo |

---

##  Technologies Used

- **Python** - Training, export, and web app
- **PyTorch & Torchvision** - Model training (MobileNetV3 Small)
- **Streamlit** - Interactive web interface
- **LibTorch** - C++ inference using PyTorch C++ API
- **OpenCV** - Image loading and preprocessing in C++
- **ONNX** - Cross-platform model export

---

## Project Structure

```
animals10-classifier/
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── models/
│   ├── mobilenetv3_animals10.pth       # Trained PyTorch model
│   ├── mobilenetv3_scripted.pt         # TorchScript export for C++
│   └── mobilenetv3.onnx                # ONNX export
├── src/
│   ├── prepare_data.py                 # Dataset preparation
│   ├── train.py                        # Model training
│   └── predict.py                      # Python inference
├── animals_cpp/
│   ├── predict.cpp                     # C++ inference code
│   ├── CMakeLists.txt                  # CMake build config
│   └── mobilenetv3_scripted.pt         # Model for C++
├── app.py                              # Streamlit web app
├── export_model.py                     # Export to TorchScript
├── export_onnx.py                      # Export to ONNX
├── requirements.txt
└── README.md
```

---

##  Dataset

[Animals-10 Dataset](https://www.kaggle.com/datasets/alessiocorrado99/animals10) from Kaggle — 5 classes selected out of 10.

---

##  Training

- **Model:** MobileNetV3 Small (Transfer Learning)
- **Epochs:** 5
- **Optimizer:** Adam
- **Loss Function:** CrossEntropyLoss

```bash
python src/train.py
```

---

##  Python Inference

```bash
python src/predict.py image.jpg
```

---

##  Streamlit Web App

```bash
streamlit run app.py
```

---

## C++ Inference (LibTorch)

### Prerequisites

| Tool | Link |
|------|------|
| Visual Studio Community 2022 (with C++ workload) | https://visualstudio.microsoft.com |
| LibTorch CPU | https://pytorch.org |
| OpenCV 4.x | https://opencv.org |

---

### Step 1 - Export model to TorchScript

```bash
python export_model.py
```

This generates `models/mobilenetv3_scripted.pt`.

---

### Step 2 - Build C++ project

```bash
cd animals_cpp
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=E:/libtorch
cmake --build . --config Release
```

---

### Step 3 - Run C++ inference

```bash
predict.exe E:\test.jpeg
```

---

### Example Output

```
Model loaded successfully!
Loading image: E:\test.jpeg

=== Results ===
Predicted: cat
Confidence: 75.316%

All probabilities:
  butterfly: 24.6803%
  cat: 75.316%
  dog: 0.00366021%
  elephant: 6.93002e-05%
  horse: 1.79121e-05%
```

---

## ONNX Export

Export the model to ONNX format for cross-platform deployment:

```bash
python export_onnx.py
```

This generates `models/mobilenetv3.onnx` which can be used with ONNX Runtime on any platform.

---

## Workflow Overview

```
Training (Python)
      ↓
mobilenetv3_animals10.pth
      ↓
Export (TorchScript) ──→ mobilenetv3_scripted.pt ──→ C++ Inference (LibTorch)
      ↓
Export (ONNX)        ──→ mobilenetv3.onnx         ──→ Cross-platform deployment
```

---

##  Requirements

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

##  Author

Built as part of a deep learning internship project exploring model training, web deployment, and C++ inference pipelines.
```