# Sentiment Analysis of Amazon Product Reviews

![sentiment analysis](img/sentiment_emoji.jpg)

Sentiment analysis, which is also known as opinion mining, is one of the major tasks of NLP. It studies people’s sentiments towards certain entities and detects polarity (e.g. a positive or negative opinion) within text. Sentiment analysis is often used in business to detect sentiment in social media, evaluate brand reputation and understand customers.

This project focuses on predicting sentiment of Amazon product reviews from Toys & Games category. Each review has the score that was used as a proxy of sentiment.

------

**Steps of running a sentiment analysis of product reviews:**

1. [Text preprocessing & creating word vectors](https://github.com/klaudia-nazarko/nlp-sentiment-analysis/blob/main/nlp_text_preprocessing.ipynb) - transform text into word vectors with various methods: Bag of Words, TF-IDF, Co-occurrence Matrix with SVD & NMF decomposition, Word2Vec
2. Sentiment Analysis with Machine Learning models:
   - [comparison of ML models](https://github.com/klaudia-nazarko/nlp-sentiment-analysis/blob/main/sentiment_analysis_1_ml_models.ipynb) such as Naive Bayes, SGD, Logistic Regression
   - [optimising performance of ML models](https://github.com/klaudia-nazarko/nlp-sentiment-analysis/blob/main/sentiment_analysis_2_ml_optimisation.ipynb) by testing different text representations and tuning hyperparameters
3. [Sentiment Analysis with Deep Learning models](https://github.com/klaudia-nazarko/nlp-sentiment-analysis/blob/main/sentiment_analysis_3_deep_learning_models.ipynb) - comparison of performance of different neural networks: densely connected, LSTM, CNN.

------

**Reference:**

1. https://jmcauley.ucsd.edu/data/amazon/ - product reviews dataset
2. https://www.fast.ai/2019/07/08/fastai-nlp/ - Fast.ai NLP course
3. Chollet, F. (2018). *Deep Learning with Python* Manning Publications