
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load our already-trained model
model = tf.keras.models.load_model("lstm_earlystop_model.keras")

# Load the same vocabulary used during training
word_index = imdb.get_word_index()

# Convert a review into numerical IDs
def preprocess_review(review):
    words = review.lower().split()

    sequence = [1]  # Start-of-review token

    for word in words:
        word_id = word_index.get(word, 2) + 3 if word in word_index else 2

        if word_id >= 10000:
            word_id = 2

        sequence.append(word_id)

    # Keep the same input length used during training
    sequence = pad_sequences(
        [sequence],
        maxlen=200,
        padding="post",
        truncating="post"
    )

    return sequence


# Enter a movie review
review = input("Enter a movie review: ")

# Convert text into numbers
processed_review = preprocess_review(review)

# Predict sentiment
prediction = model.predict(processed_review, verbose=0)[0][0]

if prediction >= 0.5:
    print("Sentiment: Positive 😊")
else:
    print("Sentiment: Negative 😞")

print("Prediction score:", prediction)