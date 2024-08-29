import keras
from keras.models import load_model
import streamlit as st 
import tensorflow as tf
import numpy as np
from PIL import Image
import io

st.header('Flower Classification CNN Model')
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

model = load_model('Flower_Recog_Model.keras')

def classify_images(image):
    input_image = image.resize((180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = 'The Image belongs to ' + flower_names[np.argmax(result)] + ' with a score of ' + str(np.max(result) * 100)
    return outcome

uploaded_file = st.file_uploader('Upload an Image')
if uploaded_file is not None:
    # Use PIL to open the image file directly
    image = Image.open(io.BytesIO(uploaded_file.read()))

    st.image(image, width=200)

    st.markdown(classify_images(image))
