# Sentiment Analysis Using Twitter Data

This code provides a sentiment analysis of Twitter data using various machine learning algorithms. It includes data loading, data visualizations, text normalization, finding missing values, and model evaluation.

## Libraries Required

The following libraries are required to run this code:

- pandas
- numpy
- random
- colorama
- IPython.display
- matplotlib
- seaborn
- wordcloud
- plotly
- missingno
- re
- emoji
- contractions
- nltk
- warnings
- string
- sklearn
- xgboost

Make sure to install these libraries before running the code.

## Data Loading

The code reads the datasets using the pandas library.

## Data Visualizations

This section includes visualizations of the training data, such as sentiment count, missing values, and word clouds based on sentiment.

## Text Normalization

The code includes a function for cleaning tweets by removing URLs, mentions, hashtags, punctuation, and stop words. It also performs tokenization, stemming, and lemmatization to normalize the text.

## Finding Missing Values

The code uses the `missingno` library to visualize missing values in the training data.

## Non-Normalized Text to Normalized Text

The code applies the `cleaning_tweets` function to the text data in the training dataset to obtain normalized tokens.

## Word Relation Sentiment

This section calculates the frequency of words in different sentiment categories (positive, neutral, negative) and prints the results.

## Using CountVectorizer and Tf-Idf

The code defines functions for fitting the CountVectorizer and Tf-Idf vectorizers. These vectorizers are then used to transform the training and testing data.

## Testing a Tweet

The code includes a function to predict the sentiment of a given tweet using a trained model. The function cleans and transforms the tweet and returns the predicted sentiment.

## Model Evaluation

This section evaluates the performance of different machine learning models using the Naive Bayes classifier and Bernoulli Naive Bayes classifier. It calculates the accuracy, generates a confusion matrix, and prints a classification report for each model.

Make sure to provide the necessary input data and execute the code to perform sentiment analysis on Twitter data.
