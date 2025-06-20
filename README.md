# Real-Time Sentiment Tracking of 2024 U.S. Elections

This project uses Natural Language Processing (NLP) to perform real-time sentiment analysis on tweets related to the 2024 U.S. presidential elections. It classifies tweets into **positive**, **negative**, or **neutral** sentiments using both traditional machine learning models and deep learning (LSTM) approaches.

## Highlights

- **Data Collection**: Tweets were scraped using relevant election hashtags (e.g., #USElections, #Vote2024).
- **Preprocessing**: URL, mentions, emojis, and punctuation removal; tokenization; lemmatization; stopword filtering.
- **Modeling Approaches**:
  - **Logistic Regression & Naive Bayes** with TF-IDF
  - **LSTM Neural Network** with Word2Vec-style embeddings
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score
- **Best Performance**: LSTM model with ~0.81 F1-score
- **Visualization**: Trend analysis of public sentiment over time using Matplotlib and Plotly

## Tech Stack

- Python (Pandas, NumPy, Scikit-learn, Keras, Matplotlib, Seaborn, Plotly)
- NLP Libraries: NLTK, SpaCy
- Embeddings: GloVe (pre-trained)
- Deep Learning: LSTM via Keras

## Outcome

The project shows how sentiment evolves in response to political events like debates or rallies, offering actionable insights for political analysts, journalists, and researchers.
