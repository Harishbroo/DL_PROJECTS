
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=10000
)

# 2. Pad reviews
X_train = pad_sequences(
    X_train, maxlen=200, padding="post", truncating="post"
)
X_test = pad_sequences(
    X_test, maxlen=200, padding="post", truncating="post"
)

# 3. Build the same LSTM architecture
model = Sequential([
    tf.keras.Input(shape=(200,)),
    Embedding(10000, 32, mask_zero=True),
    LSTM(32),
    Dense(1, activation="sigmoid")
])

# 4. Compile
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 5. Configure EarlyStopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)
# EarlyStopping = Stops training when improvement stalls.
# monitor="val_loss" = Watches validation loss after each epoch.
# patience=2 =  Waits two consecutive epochs without improvement before stopping.
# restore_best_weights=True =  Restores the weights from the epoch with the lowest validation loss.

# 6. Train with EarlyStopping
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)

# 7. Evaluate on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("EarlyStopping Test Loss:", test_loss)
print("EarlyStopping Test Accuracy:", test_accuracy)

# 8. Save this experiment separately
model.save("lstm_earlystop_model.keras")