# 🔢 Next Number Prediction using SimpleRNN

## 📖 Overview

This project demonstrates how a **Recurrent Neural Network (RNN)** can learn sequential patterns and predict the next number in a sequence.

A SimpleRNN model is built using TensorFlow and Keras. It learns from small numerical sequences and predicts the next number based on the previous three numbers.

For example:

- Input: `[1, 2, 3]` → Output: `4`
- Input: `[2, 3, 4]` → Output: `5`
- Input: `[3, 4, 5]` → Output: `6`

The project introduces the basic concept of sequence learning using recurrent neural networks.

---

## 📁 Project Files

- `rnn_next_number.py` – Contains sequence preparation, SimpleRNN model creation, training, and prediction.

---

## ⚙️ Workflow

1. **Data Preparation** – Create numerical input sequences and their corresponding next-number targets.
2. **Input Reshaping** – Represent the data in three-dimensional format required by the RNN.
3. **Model Building** – Create a SimpleRNN model with 8 hidden units.
4. **Model Compilation** – Configure the model using Adam optimizer and Mean Squared Error loss.
5. **Model Training** – Train the model for 1,000 epochs.
6. **Training Evaluation** – Display the final training loss and compare actual and predicted values.
7. **New Prediction** – Provide a new sequence `[5, 6, 7]` and predict the next number.

---

## 📊 Dataset Preparation

The project uses a small manually created dataset.

**Input sequences (X):**

```python
X = np.array([
    [[1], [2], [3]],
    [[2], [3], [4]],
    [[3], [4], [5]],
    [[4], [5], [6]]
], dtype=np.float32)
```

**Target values (y):**

```python
y = np.array([4, 5, 6, 7], dtype=np.float32)
```

Each input contains three consecutive numbers, and its target is the next number.

### Understanding the Input Shape

The input shape is:

`(4, 3, 1)`

- `4` – Number of sequences (samples).
- `3` – Number of time steps in each sequence.
- `1` – Number of features at each time step.

RNN models expect sequential input in the format:

`(samples, time_steps, features)`

---

## 🧠 Model Architecture

The project uses a **SimpleRNN** layer followed by a Dense output layer.

```python
model = Sequential([
    SimpleRNN(8, activation="tanh", input_shape=(3, 1)),
    Dense(1)
])
```

### Architecture Explanation

| Layer | Purpose |
|---|---|
| SimpleRNN(8) | Learns patterns from the input sequence using 8 recurrent units. |
| `tanh` | Activation function used inside the recurrent layer. |
| Dense(1) | Produces a single numerical prediction. |

### Model Compilation

```python
model.compile(
    optimizer="adam",
    loss="mse"
)
```

- **Adam:** Optimizer that updates model weights during training.
- **MSE (Mean Squared Error):** Measures the difference between actual and predicted values.

---

## 🚀 Model Training

The model is trained using:

```python
model.fit(X, y, epochs=1000, verbose=0)
```

- `X` – Input sequences.
- `y` – Expected next numbers.
- `epochs=1000` – The model goes through the training dataset 1,000 times.
- `verbose=0` – Hides training progress output.

The final training loss is displayed after training.

---

## 🔍 Making Predictions

### 1. Predictions on Training Data

```python
train_predictions = model.predict(X, verbose=0)

print("Actual:", y)
print("Predicted:", train_predictions.flatten())
```

This compares the model's predictions with the actual target values.

### 2. Prediction on a New Sequence

```python
new_input = np.array(
    [[[5], [6], [7]]],
    dtype=np.float32
)

prediction = model.predict(new_input, verbose=0)

print("Pred
