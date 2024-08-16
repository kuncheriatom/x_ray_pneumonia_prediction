import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing import image
import numpy as np

# Function to preprocess the image
def preprocess_image(img_path, img_height, img_width, model_type="CNN"):
    # Load the image and convert to grayscale
    img = image.load_img(img_path, target_size=(img_height, img_width), color_mode='grayscale')
    img_array = image.img_to_array(img)

    # Normalize the image array
    img_array = img_array / 255.0  # Normalize for all models

    if model_type in ["Logistic Regression", "Decision Tree"]:
        img_array = img_array.flatten()  # Flatten for Logistic Regression and Decision Tree
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    else:
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension for CNN and ANN

    return img_array

# Load the Keras model
@st.cache_resource
def load_keras_model():
    return tf.keras.models.load_model('best_model.keras')

# Load the other models
@st.cache_resource
def load_pickle_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# Define model paths and validation accuracies
models_info = {
    "ANN Model": {
        "path": "ann_model.pkl",
        "accuracy": 71.0  
    },
    "Decision Tree": {
        "path": "decision_tree_classifier_model.pkl",
        "accuracy": 71.96
    },
    "Logistic Regression": {
        "path": "logistic_regression_model.pkl",
        "accuracy": 71.47
    },
    "CNN Model": {
        "path": "best_model.keras",
        "accuracy": 86.21
    }
}

# Streamlit UI
st.title("X-ray Image Classification")
st.write("Upload an X-ray image to classify it as Normal or Pneumonia.")

# Model selection
model_name = st.selectbox("Choose a model:", list(models_info.keys()))

# Display selected model accuracy
st.write(f"Selected Model: {model_name}")
st.write(f"Validation Accuracy: {models_info[model_name]['accuracy']:.2f}%")

# Load the selected model
if model_name == "CNN Model":
    model = load_keras_model()
else:
    model = load_pickle_model(models_info[model_name]["path"])

# File uploader for image
uploaded_file = st.file_uploader("Choose an X-ray image...", type="jpeg")

if uploaded_file is not None:
    with open("temp.jpeg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Use the appropriate preprocessing for the selected model
    img_height, img_width = 224, 224  # Use the same dimensions as used during training
    preprocessed_img = preprocess_image(
        "temp.jpeg", 
        img_height, 
        img_width,
        model_type=model_name  # Pass the model name directly
    )
    
    st.image(uploaded_file, caption="Uploaded X-ray Image", use_column_width=True)
    
    # Prediction logic
    prediction = model.predict(preprocessed_img)
    prediction_label = "Pneumonia" if prediction[0] > 0.5 else "Normal"

    # Highlight the prediction
    st.markdown(f"**Prediction:** <span style='color: {'red' if prediction_label == 'Pneumonia' else 'green'}; font-size: 24px;'>{prediction_label}</span> (Model: {model_name})", unsafe_allow_html=True)
