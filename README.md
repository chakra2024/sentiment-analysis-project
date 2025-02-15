Emotion Analysis using Deep Learning

📌 Project Overview

This project focuses on predicting emotions from textual data using a deep learning model. A Bidirectional LSTM network was implemented to classify input text into six emotional categories: anger, fear, joy, love, sadness, and surprise. The dataset was preprocessed using NLP techniques to improve model accuracy and generalization.

📌 Dataset - emotions_dataset (uploaded in Zip format) contains sentences and emotions/sentiments like - Joy, Anger, Sadness etc.

✨ Features

Text Preprocessing: Tokenization, stopword removal, lemmatization, and padding.

Neural Network Model: A Bidirectional LSTM with dropout and dense layers for classification.

Performance Optimization: Implemented early stopping, class weighting, and learning rate scheduling.

Evaluation Metrics: Achieved 92.37% validation accuracy, using precision, recall, and F1-score to assess performance.

Visualization: Confusion matrix and training history plots to analyze the model’s effectiveness.

⚙️ Prerequisites

Ensure you have the following installed:

Python: 3.10 or above

TensorFlow: 2.x

Keras

NLTK

Scikit-learn

Pandas

NumPy

Matplotlib

Seaborn

🚀 Installation

Clone the repository:

git clone <repository_link>

Install dependencies:

pip install -r requirements.txt

Run the training script:

Sentiment Analysis - DL and NLP -Soham.ipynb

📊 Results

Achieved 92.37% validation accuracy with an optimized Bidirectional LSTM architecture.

The model effectively classified emotions with high precision and recall, particularly for joy and sadness.

Identified minor class imbalances in underrepresented emotions such as surprise.

🔮 Future Enhancements

Implement pre-trained embeddings like GloVe for improved word representation.

Apply data augmentation techniques to balance class distribution.

Deploy as a REST API for real-time emotion prediction.

🛠 Example Usage

from model import predict_emotion
text = "I am feeling so happy today!"
print(predict_emotion(text))
# Output: Joy

📜 License

This project is open-source and available under the MIT License.

