# Disaster Tweet Analysis

This project focuses on the analysis and classification of tweets to determine if they are related to disasters or not. The primary goal is to build a machine learning model that can effectively identify disaster-related tweets using natural language processing techniques.

## Introduction

The purpose of this project is to develop a model that can accurately classify tweets as disaster-related or not, using a dataset containing disaster and non-disaster tweets. Natural language processing techniques are applied throughout the analysis.

## Dataset

The dataset consists of disaster and non-disaster tweets, with columns 'id', 'keyword', 'location', and 'text'. The 'target' column is the binary label indicating whether the tweet is related to a disaster (1) or not (0).

## Data Visualization

Visualizations such as countplots, bar charts, word clouds, and heatmaps are used to explore the dataset and gain insights into the distribution of disaster and non-disaster tweets, as well as the most frequent words and the relationships between features.

## Data Preprocessing

The text data is cleaned by removing URLs, mentions, hashtags, and stopwords. Additionally, the text is tokenized, and lemmatization is applied to reduce words to their base forms.

## Feature Engineering

Features such as text length, word count, and the frequency of specific words are extracted. The 'keyword' column is target-encoded, and TfidfVectorizer is used to transform the text data into numerical features.

## Model Training

A logistic regression model is trained on the processed dataset, using the engineered features to predict whether a tweet is related to a disaster or not.

## Model Evaluation

The performance of the logistic regression model is evaluated using cross-validation and F1 scores, which provide insights into the model's effectiveness in classifying tweets.

## Feature Selection

Recursive feature elimination with cross-validation (RFECV) is applied to identify the optimal number of features for the model, improving its performance and reducing the risk of overfitting.

## Conclusion

By leveraging natural language processing techniques and machine learning, this project demonstrates the feasibility of building a model that can accurately classify disaster-related tweets. Further improvements and extensions can be explored to enhance the model's performance and adapt it for real-time applications.

## Getting Started

### Prerequisites

The following Python libraries are required for this project:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- nltk
- wordcloud

You can install these libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud

