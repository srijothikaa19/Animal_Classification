# Animals-10 Image Classification using PyTorch and Streamlit

## Project Overview

This project is an image classification application built using PyTorch, MobileNetV3 Small, and Streamlit.

The model is trained on five classes from the Animals-10 dataset:

* Butterfly
* Cat
* Dog
* Elephant
* Horse

The application allows users to upload an image and receive a predicted class along with confidence score.

---

## Technologies Used

* Python
* PyTorch
* Torchvision
* Streamlit
* PIL (Pillow)
* Scikit-learn
* MobileNetV3

---

## Dataset

Animals-10 Dataset (Kaggle)

Selected Classes:

* gatto → cat
* cane → dog
* cavallo → horse
* elefante → elephant
* farfalla → butterfly

---

## Project Structure

animals10-classifier/

* data/

  * train/
  * val/
  * test/
* models/

  * mobilenetv3_animals10.pth
* src/

  * prepare_data.py
  * train.py
  * predict.py
* screenshots/
* app.py
* requirements.txt
* README.md

---

## Training

Model: MobileNetV3 Small (Transfer Learning)

Epochs: 5

Optimizer: Adam

Loss Function: CrossEntropyLoss

---

## Run Training

python src/train.py

---

## Run Prediction

python src/predict.py image.jpg

---

## Run Streamlit App

streamlit run app.py

---

## Results

The model successfully classifies:

* Butterfly
* Cat
* Dog
* Elephant
* Horse

and displays prediction confidence scores through a Streamlit web interface.