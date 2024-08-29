
# Flower Recognition App

This repository contains a Streamlit web application and the model training code for recognizing different types of flowers. The app allows users to upload an image of a flower, and it will predict the flower's type using a pre-trained model.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Streamlit App](#streamlit-app)
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

## Overview

This project aims to classify flower images into five different categories: daisy, dandelion, rose, sunflower, and tulip. The model is trained using a Convolutional Neural Network (CNN) implemented in TensorFlow/Keras.

The repository contains:
- `Flower_recog_Model.ipynb`: A Jupyter notebook for training the CNN model.
- `app.py`: A Streamlit app to interact with the trained model for flower recognition.

## Dataset

The dataset used for training the model is the [Flowers Recognition Dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) from Kaggle. It contains images of flowers belonging to the following five categories:
- Daisy
- Dandelion
- Rose
- Sunflower
- Tulip

## Model Training

The model training process is described in the `Flower_recog_Model.ipynb` notebook. It covers:
1. **Data Preprocessing**: Loading and preprocessing the flower images.
2. **Model Architecture**: Designing a CNN model for image classification.
3. **Training**: Training the model with the flower dataset.
4. **Evaluation**: Evaluating the model's performance on a validation set.
5. **Saving the Model**: Saving the trained model for later use in the Streamlit app.

## Streamlit App

The `app.py` file contains the code for the Streamlit web application. The app allows users to upload an image of a flower and get a prediction of the flower's type based on the trained model.

### Key Features:
- **Image Upload**: Users can upload an image of a flower.
- **Prediction**: The app will predict the flower type and display the result.
- **User-Friendly Interface**: The app is easy to use and accessible via a web browser.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/flower-recognition-app.git
    cd flower-recognition-app
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset from Kaggle and place it in the appropriate directory.

4. Train the model (optional):
   - You can skip this step if you want to use the pre-trained model.
   - Run the `Flower_recog_Model.ipynb` notebook to train the model.

5. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Usage

- To use the app, navigate to the URL provided by Streamlit after running the `app.py` file.
- Upload a flower image, and the app will predict the type of flower.

## Acknowledgments

- The flower images used for training are from the [Flowers Recognition Dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) by Alexander Mamaev.
- This project is built using [Streamlit](https://streamlit.io/) and [TensorFlow](https://www.tensorflow.org/).

##App
Demo: https://flower-image-recognition-app.onrender.com/
