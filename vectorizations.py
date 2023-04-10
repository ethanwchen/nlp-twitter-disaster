import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def vectorize_data(train, test):
    # CountVectorizer for links, mentions, and hashtags
    vec_links = CountVectorizer(min_df=5, analyzer='word', token_pattern=r'https?://\S+')
    ...
    
    return train, test

def tfidf_vectorize(train, test):
    # TfidfVectorizer for clean_text
    vec_text = TfidfVectorizer(min_df=10, ngram_range=(1, 2), stop_words='english')
    ...
    
    return train, test
