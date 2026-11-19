# Dog vs Cat Classification using CNN

## Project Overview
This project uses a **Convolutional Neural Network (CNN)** to classify images into two categories: **Dog** and **Cat**.

The model is trained to learn visual features from images and predict whether an uploaded image contains a dog or a cat.

## Model Architecture
The CNN includes:
- Convolutional layers with 32, 64, and 128 filters
- Batch Normalization
- Max Pooling layers
- Fully Connected (Dense) layers
- Dropout for regularization
- Sigmoid output layer for binary classification

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Streamlit
- Jupyter Notebook

## Project Structure

```text
1. DogvsCat_CNN/
│
├── DogvsCat.ipynb       # Model training and experiments
├── app.py               # Streamlit web application
├── requirements.txt     # Project dependencies
├── Dockerfile           # Docker configuration
└── dog_cat_model.keras  # Trained CNN model (not included in GitHub)
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Harishbroo/DL_PROJECTS.git
```

### 2. Navigate to the project folder

```bash
cd DL_PROJECTS/"1. DogvsCat_CNN"
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the trained model

Place `dog_cat_model.keras` inside this folder.

**Note:** The trained model is not included in this GitHub repository because it exceeds GitHub's 100 MB individual file limit.

### 5. Run the application

```bash
streamlit run app.py
```

## What I Learned
- Building a CNN for image classification
- Understanding convolution and pooling layers
- Training and saving deep learning models
- Using OpenCV for image processing
- Deploying a deep learning model with Streamlit
