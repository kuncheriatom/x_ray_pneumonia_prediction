import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Function to preprocess the image
def preprocess_image(img_path, img_height, img_width):
    # Load the image in grayscale mode
    img = image.load_img(img_path, target_size=(img_height, img_width), color_mode='grayscale')
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    # Expand dimensions to match the model's input shape
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize the image
    img_array = img_array / 255.0  
    return img_array

# Load the best model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('best_model_.keras')

model = load_model()

# Streamlit UI
st.title("X-ray Image Classification")
st.write("Upload an X-ray image to classify it as Normal or Pneumonia.")

# File uploader for image
uploaded_file = st.file_uploader("Choose an X-ray image...", type="jpeg")

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with open("temp.jpeg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Preprocess the image
    img_height, img_width = 224, 224  # Use the same dimensions as used during training
    preprocessed_img = preprocess_image("temp.jpeg", img_height, img_width)
    
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded X-ray Image", use_column_width=True)
    
    # Make predictions
    prediction = model.predict(preprocessed_img)
    
    # Output the prediction
    if prediction[0] > 0.5:
        st.write("Prediction: Pneumonia")
    else:
        st.write("Prediction: Normal")
