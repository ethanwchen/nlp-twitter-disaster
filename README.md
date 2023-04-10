# Disaster Tweet Analysis 🌋

This project focuses on the analysis and classification of tweets to determine if they are related to disasters or not. The primary goal is to build a machine learning model that can effectively identify disaster-related tweets using natural language processing techniques.

The task belongs to a Kaggle contest called "Real or Not? NLP with Disaster Tweets". The challenge is to create a program that can accurately determine whether a tweet is about a real disaster or not.

## Introduction

The purpose of this project is to develop a model that can accurately classify tweets as disaster-related or not, using a dataset containing disaster and non-disaster tweets. Natural language processing techniques are applied throughout the analysis.

In this analysis, the dataset containing disaster and non-disaster tweets is explored through various visualizations, including countplots, bar charts, word clouds, and heatmaps. The data is preprocessed and cleaned by removing URLs, mentions, hashtags, and stopwords. Then, features such as text length, word count, and the frequency of specific words are extracted. The dataset is further processed by applying target encoding on the 'keyword' column and using TfidfVectorizer for text features. A logistic regression model is trained, and its performance is evaluated using cross-validation and F1 scores. Feature selection is performed using RFECV to determine the optimal number of features for the model.

## Dataset

The dataset consists of disaster and non-disaster tweets, with columns 'id', 'keyword', 'location', and 'text'. The 'target' column is the binary label indicating whether the tweet is related to a disaster (1) or not (0).

## Data Visualization

Visualizations such as countplots, bar charts, word clouds, and heatmaps are used to explore the dataset and gain insights into the distribution of disaster and non-disaster tweets, as well as the most frequent words and the relationships between features.

Some example visualizations generated.

![download-2](https://user-images.githubusercontent.com/96222805/230750383-ea065856-0a56-4433-ba5e-a0cacb5fbf28.png)
![download-3](https://user-images.githubusercontent.com/96222805/230750384-fba38faf-898b-4afe-9fe7-fe3b9865ed1c.png)
![download-4](https://user-images.githubusercontent.com/96222805/230750385-5b96969b-93fa-4771-9961-4893bb5fd28d.png)


## Data Preprocessing

We need to perform data cleaning on the text data in a df to standardize its format and remove any extraneous information that may not be relevant to the analysis. This is important because it helps to ensure that the machine learning models are working with consistent and reliable inputs.

The text data is cleaned by removing URLs, mentions, hashtags, and stopwords. Additionally, the text is tokenized, and lemmatization is applied to reduce words to their base forms.

By doing so, we are improving the quality of the data and making it easier to analyze and interpret, which ultimately leads to more accurate and reliable results in natural language processing tasks such as sentiment analysis, classification, and topic modeling.

Here's an example of the function runnning on sample text.

<img width="619" alt="Screen Shot 2023-04-08 at 6 56 58 PM" src="https://user-images.githubusercontent.com/96222805/230750313-7382cecc-e641-4a32-8fa5-7e61e77050a8.png">

## Feature Engineering

Features such as text length, word count, and the frequency of specific words are extracted. The 'keyword' column is target-encoded, and TfidfVectorizer is used to transform the text data into numerical features.

By creating additional features like tweet length, word count, stopword count, count of mentions, links, hashtags, punctuation, and count of uppercase letters, we can extract more information from the text data. These features may help in improving the predictive power of the machine learning models and can provide insights into the structure and characteristics of the tweets.

We also fill in the missing values in the "keyword" column using a BERT model for text analysis to provide relevant information to machine learning models that can improve their performance in predicting whether a given tweet is about a real disaster or not.

## Model Training

A logistic regression model is trained on the processed dataset, using the engineered features to predict whether a tweet is related to a disaster or not.

## Model Evaluation

The performance of the logistic regression model is evaluated using cross-validation and F1 scores, which provide insights into the model's effectiveness in classifying tweets.

Recursive feature elimination with cross-validation (RFECV) is applied to identify the optimal number of features for the model, improving its performance and reducing the risk of overfitting.

## Conclusion

By leveraging natural language processing techniques and machine learning, this project demonstrates the feasibility of building a model that can accurately classify disaster-related tweets. Further improvements and extensions can be explored to enhance the model's performance and adapt it for real-time applications.

The following Python libraries were used for this project:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- nltk
- wordcloud

You can install these libraries using pip to try running the notebook on your own.:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud
```

#### Data available here:

https://www.kaggle.com/c/nlp-getting-started/data

