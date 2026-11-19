# 🎬 Movie Review Sentiment Analysis using RNN and LSTM

## 📖 Overview

This project uses **Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM)** models to classify movie reviews as either **Positive** or **Negative**.

The project uses the IMDB movie review dataset provided by TensorFlow/Keras. Text reviews are converted into numerical sequences and padded to a fixed length before being passed to the models.

Different experiments are implemented using SimpleRNN and LSTM architectures. An additional LSTM model uses EarlyStopping to monitor validation loss and restore the best-performing weights.

A **Streamlit web application** allows users to enter their own movie reviews and receive sentiment predictions.

---

## 📊 Dataset

- **Dataset:** IMDB Movie Reviews
- **Source:** TensorFlow / Keras
- **Problem Type:** Binary Text Classification
- **Classes:** Positive and Negative
- **Vocabulary Size:** 10,000 most frequent words
- **Maximum Sequence Length:** 200 tokens
- **Training and Testing:** Predefined training and testing sets provided by Keras

---

## 📁 Project Files

- `sentiment_analysis.py` – Implements a SimpleRNN model for sentiment classification.
- `sentiment_model.keras` – Saved SimpleRNN model.
- `lstm_sentiment.py` – Implements an LSTM model for sentiment classification.
- `lstm_sentiment_model.keras` – Saved LSTM model.
- `lstm_earlystop.py` – Implements an LSTM model with EarlyStopping.
- `lstm_earlystop_model.keras` – Saved LSTM model with EarlyStopping.
- `check_model.py` – Evaluates the saved SimpleRNN model on the test dataset.
- `predict.py` – Command-line script for predicting sentiment from a movie review.
- `app.py` – Streamlit web application for interactive sentiment prediction.
- `Dockerfile` – Configuration for containerizing the Streamlit application.
- `requirements.txt` – Python dependencies.

---

## ⚙️ Workflow

1. **Dataset Loading** – Load the IMDB movie review dataset using Keras.
2. **Text Encoding** – Represent reviews as numerical sequences using the dataset's word vocabulary.
3. **Padding** – Make all review sequences exactly 200 tokens long.
4. **Model Building** – Create SimpleRNN and LSTM architectures.
5. **Model Compilation** – Configure models using Adam optimizer and Binary Crossentropy loss.
6. **Model Training** – Train the models using the training dataset.
7. **EarlyStopping** – Train an additional LSTM model with EarlyStopping to monitor validation loss.
8. **Model Evaluation** – Evaluate the models on unseen test data.
9. **Model Saving** – Save the trained models in Keras format.
10. **Prediction** – Convert new movie reviews into numerical sequences and predict their sentiment.
11. **Streamlit Application** – Build a web interface for users to enter reviews and view predictions.
12. **Docker Containerization** – Package the application, dependencies, and selected trained model.

---

## 🧠 Model Architectures

### 1. SimpleRNN

The first experiment uses a SimpleRNN model.

```python
model = Sequential([
    tf.keras.Input(shape=(200,)),
    Embedding(input_dim=10000, output_dim=32, mask_zero=True),
    SimpleRNN(32),
    Dense(1, activation="sigmoid")
])
```

**Architecture:**

- Input layer: 200 tokens
- Embedding layer: 10,000 vocabulary size and 32-dimensional embeddings
- SimpleRNN layer: 32 units
- Dense output layer: 1 neuron with sigmoid activation

### 2. LSTM

The second experiment replaces SimpleRNN with an LSTM layer.

```python
model = Sequential([
    tf.keras.Input(shape=(200,)),
    Embedding(input_dim=10000, output_dim=32, mask_zero=True),
    LSTM(32),
    Dense(1, activation="sigmoid")
])
```

**Architecture:**

- Input layer: 200 tokens
- Embedding layer: 10,000 vocabulary size and 32-dimensional embeddings
- LSTM layer: 32 units
- Dense output layer: 1 neuron with sigmoid activation

### 3. LSTM with EarlyStopping

An additional LSTM model uses EarlyStopping to monitor validation loss during training.

```python
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)
```

- `monitor="val_loss"` – Monitors validation loss after each epoch.
- `patience=2` – Waits for two consecutive epochs without improvement before stopping.
- `restore_best_weights=True` – Restores the weights from the epoch with the lowest validation loss.

The model is trained with a maximum of 10 epochs, but EarlyStopping can stop training earlier if validation loss stops improving.

---

## 🔄 Text Preprocessing

Before feeding movie reviews into the models, the text must be converted into numerical sequences.

### 1. Vocabulary

The IMDB dataset uses a word index to represent words as integer IDs.

```python
(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=10000
)
```

`num_words=10000` keeps the most frequent 10,000 words.

### 2. Padding

Reviews have different lengths, so padding makes every sequence the same length.

```python
X_train = pad_sequences(
    X_train,
    maxlen=200,
    padding="post",
    truncating="post"
)
```

- `maxlen=200` – Each review contains exactly 200 tokens.
- `padding="post"` – Adds zeros at the end of shorter reviews.
- `truncating="post"` – Removes tokens from the end of longer reviews.

### 3. Embedding Layer

The Embedding layer converts integer word IDs into dense numerical vectors.

```python
Embedding(
    input_dim=10000,
    output_dim=32,
    mask_zero=True
)
```

- `input_dim=10000` – Vocabulary size.
- `output_dim=32` – Each word is represented by a 32-dimensional vector.
- `mask_zero=True` – Allows the model to ignore padded zero values.

---

## 🚀 Model Training and Evaluation

The models use:

| Configuration | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Evaluation Metric | Accuracy |
| Batch Size | 32 |
| Maximum Sequence Length | 200 |
| Vocabulary Size | 10,000 |

Training configurations:

| Model | Epochs | EarlyStopping |
|---|---:|---|
| SimpleRNN | 5 | No |
| LSTM | 5 | No |
| LSTM with EarlyStopping | Up to 10 | Yes |

The models are evaluated using the test dataset to calculate test loss and accuracy.

The early-stopping model is saved as `lstm_earlystop_model.keras` and is used by the Streamlit application.

---

## 🔍 Making Predictions

The project supports two ways to predict sentiment.

### 1. Command-Line Prediction

The `predict.py` script allows users to enter a movie review directly in the terminal.

```python
prediction = model.predict(
    processed_review,
    verbose=0
)[0][0]

if prediction >= 0.5:
    print("Sentiment: Positive")
else:
    print("Sentiment: Negative")
```

The model produces a sigmoid score between 0 and 1.

- Score greater than or equal to 0.5: Positive
- Score below 0.5: Negative

### 2. Streamlit Web Application

The `app.py` file provides an interactive interface.

**How it works:**

1. User enters a movie review.
2. The application converts the text into numerical IDs.
3. The review is padded to 200 tokens.
4. The processed review is passed to the saved LSTM model.
5. The application displays Positive or Negative sentiment.
6. The model's prediction score is also displayed.

The application loads `lstm_earlystop_model.keras` and caches the model and vocabulary using Streamlit's `st.cache_resource`.

---

## 🛠️ Technologies Used

- Python
- NumPy
- TensorFlow / Keras
- IMDB Dataset
- SimpleRNN
- LSTM
- Embedding Layer
- EarlyStopping
- Streamlit
- Docker
- Jupyter Notebook / Python

---

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Harishbroo/DL_PROJECTS.git
```

### 2. Navigate to the project folder

```bash
cd DL_PROJECTS/"4. Sentiment Analysis"
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

Open the application at:

`http://localhost:8501`

---

## 🐳 Docker

The project includes a Dockerfile to package the Streamlit application, dependencies, and the saved EarlyStopping LSTM model.

### Build the Docker image

```bash
docker build -t sentiment-analysis .
```

### Run the Docker container

```bash
docker run -p 8501:8501 sentiment-analysis
```

Open the application at:

`http://localhost:8501`

---

## 👨‍💻 Author

**Harish Parajuli**
