
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the same test dataset
(_, _), (X_test, y_test) = imdb.load_data(num_words=10000)

# Apply the same preprocessing used during training
X_test = pad_sequences(
    X_test,
    maxlen=200,
    padding="post",
    truncating="post"
)

# Load our saved model
model = tf.keras.models.load_model("sentiment_model.keras")

# Evaluate without training
test_loss, test_accuracy = model.evaluate(
    X_test, y_test, verbose=1
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)