# Movie Review Sentiment Analysis using DistilBERT

## Project Overview
This project uses a pre-trained DistilBERT model from Hugging Face Transformers to perform sentiment analysis on movie reviews.

The model analyzes text and classifies each review as either **POSITIVE** or **NEGATIVE**.

## Features
- Uses a pre-trained DistilBERT model.
- Performs sentiment analysis on individual reviews.
- Analyzes multiple reviews in a batch.
- Displays sentiment labels and confidence scores.
- Uses Hugging Face Transformers and PyTorch.

## Model
- **Model:** DistilBERT
- **Pre-trained model:** `distilbert-base-uncased-finetuned-sst-2-english`
- **Task:** Binary Sentiment Classification
- **Labels:** POSITIVE and NEGATIVE

## Technologies Used
- Python
- Hugging Face Transformers
- PyTorch
- Pandas
- Jupyter Notebook / Google Colab

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Usage

Run the notebook:

`Movie_Review_Sentiment_Analysis.ipynb`

You can provide a movie review as input, and the model predicts its sentiment along with a confidence score.

Example:

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

result = classifier("This movie was absolutely amazing!")
print(result)
```

Example output:

```python
[{'label': 'POSITIVE', 'score': 0.9998}]
```

## Important Note
This model performs binary sentiment classification. It does not have a separate NEUTRAL category, so neutral or mixed reviews will still be classified as either POSITIVE or NEGATIVE.

## Learning Outcomes
- Understanding pre-trained Transformer models.
- Using Hugging Face pipelines.
- Exploring tokenization and model inputs.
- Understanding logits and softmax probabilities.
- Performing single and batch text predictions.
- Interpreting model confidence scores.

## Project Structure

```text
5. Movie Review Sentiment Analysis using DistilBERTHugging Face/
│
├── Movie_Review_Sentiment_Analysis.ipynb
├── requirements.txt
└── README.md
```

## Author
**Harish Parajuli**

GitHub: [Harishbroo](https://github.com/Harishbroo)
