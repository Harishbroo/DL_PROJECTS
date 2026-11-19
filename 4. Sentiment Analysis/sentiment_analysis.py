import tensorflow as tf
from tensorflow.keras.datasets import imdb

# Load the IMDB dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)
# imdb.load_data() — loads movie reviews already converted into numbers.
# num_words=10000 — keeps the most frequent 10,000 words in the vocabulary.

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("First review:", X_train[0])
print("First label:", y_train[0])

#Padding makes all reviews the same length by adding zeros to shorter reviews, so we can feed them together into the RNN in batches.
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Make all reviews the same length
X_train = pad_sequences(X_train, maxlen=200, padding="post", truncating="post")
X_test = pad_sequences(X_test, maxlen=200, padding="post", truncating="post")
# maxlen=200 — every review will contain exactly 200 tokens.
# padding="post" — add zeros at the end of shorter reviews.
# truncating="post" — if a review has more than 200 tokens, remove tokens from the end.

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
print("First padded review length:", len(X_train[0]))


##Build the model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# Build the model
model = Sequential([
    tf.keras.Input(shape=(200,)),
    Embedding(input_dim=10000, output_dim=32, mask_zero=True),
    SimpleRNN(32),
    Dense(1, activation="sigmoid")
])

# Configure the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Display model architecture
model.summary()


# Train the RNN model
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)


# Evaluate the model on unseen test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

model.save("sentiment_model.keras")