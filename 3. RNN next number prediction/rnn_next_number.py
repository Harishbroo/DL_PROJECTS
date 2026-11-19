
import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

X = np.array([
    [[1], [2], [3]],
    [[2], [3], [4]],
    [[3], [4], [5]],
    [[4], [5], [6]]
], dtype=np.float32)

y = np.array([4, 5, 6, 7], dtype=np.float32)

print("Input shape:", X.shape)
print("Target shape:", y.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

model = Sequential([
    SimpleRNN(8, activation="tanh", input_shape=(3, 1)),
    Dense(1)
])

model.summary()

##Compile the model
model.compile(
    optimizer="adam",
    loss="mse"
)

##Train the model
model.fit(X, y, epochs=1000, verbose=0)
print("Final training loss:", model.history.history["loss"][-1])
#verbose=0 — hides the training progress output.
# verbose=0 → Shows nothing.
# verbose=1 → Shows a progress bar.
# verbose=2 → Shows one line per epoch.
train_predictions = model.predict(X, verbose=0)

print("Actual:", y)
print("Predicted:", train_predictions.flatten())

# Test with a new sequence
new_input = np.array([[[5], [6], [7]]], dtype=np.float32)
prediction = model.predict(new_input, verbose=0)
print("Predicted next number:", prediction[0][0])