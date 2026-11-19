# 🐶 Dog vs Cat Classification using CNN

## 📖 Overview

This project uses a **Convolutional Neural Network (CNN)** to classify images into two categories: **Dog** and **Cat**.

The model learns visual patterns and features from images to distinguish between dogs and cats. A **Streamlit** web application allows users to upload an image and receive a prediction.

---

## 📁 Project Files

- `DogvsCat.ipynb` – Data preprocessing, CNN model building, training, and evaluation
- `app.py` – Streamlit web application for image classification
- `requirements.txt` – Python dependencies
- `Dockerfile` – Docker configuration
- `dog_cat_model.keras` – Trained CNN model (not included in this repository because it exceeds GitHub's 100 MB individual file limit)

---

## ⚙️ Workflow

- Dataset loading and preparation
- Image preprocessing and resizing
- Pixel normalization
- Building the CNN architecture
- Model training
- Model evaluation
- Saving the trained model
- Loading the model into a Streamlit application
- Predicting whether an uploaded image contains a dog or a cat

---

## 🧠 Model Architecture

The CNN architecture consists of:

- Convolutional layers with 32, 64, and 128 filters
- Batch Normalization
- Max Pooling layers
- Fully Connected (Dense) layers
- Dropout for regularization
- Sigmoid output layer for binary classification

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Streamlit
- Jupyter Notebook
- Docker

---

## 🚀 Run Locally

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

Place the trained `dog_cat_model.keras` file inside this folder.

**Note:** The model file is excluded from GitHub because it is larger than 100 MB. It must be available locally before running the application.

### 5. Run the application

```bash
streamlit run app.py
```

---

## 🐳 Docker

The project includes a Dockerfile for containerization.

Build the Docker image:

```bash
docker build -t dog-cat-cnn .
```

Run the Docker container:

```bash
docker run -p 8501:8501 dog-cat-cnn
```

---

## 👨‍💻 Author

**Harish Parajuli**
