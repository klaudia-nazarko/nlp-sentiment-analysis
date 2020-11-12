# Sentiment Analysis of Amazon Product Reviews

![sentiment analysis](img/sentiment_emoji.jpg)

Sentiment analysis, which is also known as opinion mining, is one of the major tasks of NLP. It studies people’s sentiments towards certain entities and detects polarity (e.g. a positive or negative opinion) within text. Sentiment analysis is often used in business to detect sentiment in social media, evaluate brand reputation and understand customers.

This project focuses on predicting sentiment of Amazon product reviews from Toys & Games category. Each review has the score that was used as a proxy of sentiment.

Steps of running a sentiment analysis of product reviews:

1. Text preprocessing & creating word vectors - transform text into word vectors with various methods: Bag of Words, TF-IDF, Co-occurrence Matrix with SVD & NMF decomposition, Word2Vec
2. Sentiment Analysis with Machine Learning models:
   - comparison of ML models such as Naive Bayes, SGD, Logistic Regression
   - optimising performance of ML models by testing different text representations and tuning hyperparameters
3. Sentiment Analysis with Deep Learning models - comparison of performance of different neural networks: densely connected, LSTM, CNN.