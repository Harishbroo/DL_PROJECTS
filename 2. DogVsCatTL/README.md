# 🐶 Dog vs Cat Classification using Transfer Learning

## 📖 Overview

This project uses **Transfer Learning with MobileNetV2** to classify images into two categories: **Dog** and **Cat**.

Instead of training a CNN from scratch, the project uses MobileNetV2 pretrained on ImageNet to reuse its learned visual features. The original classification head is removed and a custom classification layer is added for binary classification.

A **Streamlit** web application allows users to upload an image and receive a prediction.

The application is also containerized using **Docker**, with the workflow extending to Docker Hub and Render deployment.

---

## 📊 Dataset

- Dataset: `cats_vs_dogs`
- Source: TensorFlow Datasets
- Problem Type: Binary Image Classification
- Classes: Dog and Cat
- Train-Test Split: 80% Training / 20% Testing

---

## 📁 Project Files

- `Transfer_Learning.ipynb` – Dataset preparation, preprocessing, MobileNetV2 implementation, model training, and evaluation
- `app.py` – Streamlit web application for image classification
- `transfer_learning_dog_cat.keras` – Trained transfer learning model
- `requirements.txt` – Python dependencies
- `Dockerfile` – Docker configuration for building the application image

---

## ⚙️ Workflow

1. **Dataset Loading** – Load the Cats vs Dogs dataset using TensorFlow Datasets.
2. **Data Preparation** – Split the dataset into training and testing sets.
3. **Image Preprocessing** – Resize images to 224 × 224 and apply the required preprocessing.
4. **Transfer Learning** – Load MobileNetV2 with pretrained ImageNet weights.
5. **Feature Extraction** – Remove the original classification head and freeze the pretrained base layers.
6. **Custom Classification** – Add a custom sigmoid output layer for binary classification.
7. **Model Training** – Train the custom classification layer.
8. **Model Saving** – Save the trained model as `transfer_learning_dog_cat.keras`.
9. **Streamlit Application** – Build an interface for image uploads and predictions.
10. **Docker Containerization** – Package the application, dependencies, and trained model.
11. **Deployment Workflow** – Push the Docker image to Docker Hub and deploy the containerized application on Render.

---

## 🧠 Model Architecture

**Base Model: MobileNetV2**

- Pretrained on ImageNet
- `include_top=False` to remove the original classification head
- Input shape: `(224, 224, 3)`
- Output feature map: `(7, 7, 1280)`
- Pretrained base layers frozen during training
- Custom Dense output layer with 1 neuron
- Sigmoid activation for binary classification

The sigmoid activation produces a score between 0 and 1. The application interprets this score using the same class-label mapping used during training.

---

## 🔍 How Prediction Works

1. A user uploads a dog or cat image through the Streamlit application.
2. The application resizes and preprocesses the image.
3. The processed image is passed to the trained model.
4. The model produces a prediction score.
5. The application interprets the score and displays the predicted class: **Dog** or **Cat**.

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- TensorFlow Datasets
- MobileNetV2
- NumPy
- Streamlit
- Docker
- Docker Hub
- Render
- Jupyter Notebook

---

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Harishbroo/DL_PROJECTS.git
```

### 2. Navigate to the project folder

```bash
cd DL_PROJECTS/"2. DogVsCatTL"
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

---

## 🐳 Docker

The project includes a Dockerfile to package the Streamlit application, dependencies, and trained model.

### Build the Docker image

```bash
docker build -t dog-cat-transfer-learning .
```

### Run the Docker container

```bash
docker run -p 8501:8501 dog-cat-transfer-learning
```

Open the application at:

`http://localhost:8501`

---

## 🐳 Docker Hub and Deployment

The deployment workflow includes:

- Building the Docker image locally.
- Testing the container.
- Pushing the image to Docker Hub.
- Deploying the containerized application using Render.

Docker provides a consistent environment for running the application, while Render hosts it online.

---

## 🛠️ Common Troubleshooting

| Issue | Possible cause |
|---|---|
| `ModuleNotFoundError` | Missing Python dependency in `requirements.txt` |
| `FileNotFoundError` | Trained model missing or incorrect file path |
| Application not opening | Incorrect port or server binding configuration |
| Container crashes | Runtime error or insufficient available memory |
| Prediction errors | Incorrect image dimensions, preprocessing, or class-label mapping |

**Debugging approach:** Check the error message and deployment logs first, identify the stage where the failure occurs, and fix the underlying issue before rebuilding the Docker image.

---

## 🔮 Future Improvements

- Build a separate **FastAPI** version of the application.
- Create a `/predict` API endpoint that accepts an uploaded image.
- Return predictions as structured JSON responses.
- Reuse the existing trained model without retraining MobileNetV2.

---

## 👨‍💻 Author

**Harish Parajuli**
