# X-ray Pneumonia Prediction

This project is a Streamlit application for detecting pneumonia from chest X-ray images using different machine learning models, including a Convolutional Neural Network (CNN), Artificial Neural Network (ANN), Logistic Regression, and Decision Tree.

## Features

- **Model Selection:** Users can choose from different models to make predictions.
- **Image Upload:** Upload chest X-ray images in JPEG format.
- **Prediction Output:** The application classifies the X-ray image as either "Normal" or "Pneumonia."

## Models Used

- **CNN Model:** A deep learning model trained on X-ray images.
- **ANN Model:** An Artificial Neural Network model.
- **Logistic Regression:** A classic machine learning model for binary classification.
- **Decision Tree:** A simple yet effective machine learning model.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/kuncheriatom/x_ray_pneumonia_prediction.git
cd x_ray_pneumonia_prediction

```

### 2. Install Dependencies

Make sure you have Python 3.6 or higher installed. Then, install the required packages using pip:

```bash
pip install -r requirements.txt
```

### 3. Run the Application

To start the Streamlit application, run the following command:

```bash
streamlit run app.py
```


### 4. Access the Application

Open your web browser and go to `http://localhost:8501` to access the application.

you can also access the app from:

```bash
https://huggingface.co/spaces/mednow/pneumonia_detection
```

## Usage

1. Select the model you want to use from the dropdown menu.
2. Upload a chest X-ray image in JPEG format.
3. Click on the "Predict" button to get the classification result.

