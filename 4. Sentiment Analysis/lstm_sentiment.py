
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# 1. Load the IMDB dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=10000
)

# 2. Make every review 200 numbers long
X_train = pad_sequences(
    X_train, maxlen=200, padding="post", truncating="post"
)
X_test = pad_sequences(
    X_test, maxlen=200, padding="post", truncating="post"
)

# 3. Build the LSTM model
model = Sequential([
    tf.keras.Input(shape=(200,)),
    Embedding(input_dim=10000, output_dim=32, mask_zero=True),
    LSTM(32),
    Dense(1, activation="sigmoid")
])

# 4. Configure training
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 5. Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)

# 6. Evaluate on unseen test reviews
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("LSTM Test Loss:", test_loss)
print("LSTM Test Accuracy:", test_accuracy)

# 7. Save the trained LSTM
model.save("lstm_sentiment_model.keras")