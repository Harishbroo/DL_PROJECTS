
import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence


# 1. Configure the web page
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬"
)

st.title("🎬 Movie Review Sentiment Analysis")
st.write(
    "Enter a movie review to predict whether it is positive or negative."
)


# 2. Load the saved model and vocabulary only once
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model(
        "lstm_earlystop_model.keras"
    )
    word_index = imdb.get_word_index()

    return model, word_index


model, word_index = load_resources()


# 3. Convert a review into numerical IDs
def preprocess_review(review):
    words = text_to_word_sequence(review)

    sequence = [1]  # Start-of-review token

    for word in words:
        word_id = word_index.get(word)

        if word_id is None:
            # Unknown word
            word_id = 2
        else:
            # Match the IMDB dataset's word-index offset
            word_id = word_id + 3

            if word_id >= 10000:
                # Word is outside the 10,000-word vocabulary
                word_id = 2

        sequence.append(word_id)

    # Make the review exactly 200 tokens long
    sequence = pad_sequences(
        [sequence],
        maxlen=200,
        padding="post",
        truncating="post"
    )

    return sequence


# 4. Get a review from the user
review = st.text_area(
    "Enter your movie review:",
    placeholder="Example: This movie was amazing, with a great story..."
)


# 5. Predict sentiment when the button is clicked
if st.button("Analyze Sentiment"):
    if not review.strip():
        st.warning("Please enter a movie review first.")

    else:
        processed_review = preprocess_review(review)

        prediction = float(
            model.predict(
                processed_review,
                verbose=0
            )[0][0]
        )

        if prediction >= 0.5:
            st.success("Positive 😊")
        else:
            st.error("Negative 😞")

        st.write(f"Model score: {prediction:.4f}")

        st.caption(
            "The score is the model's sigmoid output, "
            "not a guarantee that the prediction is correct."
        )
