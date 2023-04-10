import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
import re
from wordcloud import STOPWORDS
import transformers
import torch


def clean_text(text):
    text = re.sub(r'https?://\S+', '', text)  # Remove links
    text = re.sub(r'\n', ' ', text)  # Remove line breaks
    text = re.sub('\s+', ' ', text).strip()  # Remove leading, trailing, and extra spaces
    return text


def extract_hashtags(tweet):
    return " ".join([match.group(0)[1:] for match in re.finditer(r"#\w+", tweet)]) or 'no'


def extract_mentions(tweet):
    return " ".join([match.group(0)[1:] for match in re.finditer(r"@\w+", tweet)]) or 'no'


def extract_links(tweet):
    return " ".join([match.group(0)[:] for match in re.finditer(r"https?://\S+", tweet)]) or 'no'


def process_text_data(df):
    df['clean_text'] = df['text'].apply(lambda x: clean_text(x))
    df['hashtags'] = df['text'].apply(lambda x: extract_hashtags(x))
    df['mentions'] = df['text'].apply(lambda x: extract_mentions(x))
    df['links'] = df['text'].apply(lambda x: extract_links(x))
    return df


def create_stat(df):
    df['text_len'] = df['clean_text'].apply(len)
    df['word_count'] = df["clean_text"].apply(lambda x: len(str(x).split()))
    df['stop_word_count'] = df['clean_text'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
    df['mention_count'] = df['mentions'].apply(lambda x: len(str(x).split()))
    df['link_count'] = df['links'].apply(lambda x: len(str(x).split()))
    df['hashtag_count'] = df['hashtags'].apply(lambda x: len(str(x).split()))
    df['punctuation_count'] = df['clean_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    df['caps_count'] = df['clean_text'].apply(lambda x: sum(1 for c in str(x) if c.isupper()))
    df['caps_ratio'] = df['caps_count'] / df['text_len']
    return df


def extract_keywords(text):
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    model = transformers.BertModel.from_pretrained('bert-base-uncased')

    tokens = tokenizer.encode(text, add_special_tokens=True)
    if len(tokens) == 0:
        return []
    input_ids = torch.tensor([tokens])
    with torch.no_grad():
        outputs = model(input_ids)
        embeddings = outputs.last_hidden_state[0]
    sentence_embedding = torch.sum(embeddings, dim=0)
    if len(sentence_embedding) < 3:
        num_keywords = len(sentence_embedding)
    else:
        num_keywords = 3
    keywords = []
    for i in sentence_embedding.argsort()[-num_keywords:]:
        if i >= len(tokens):
            continue
        keywords.append(tokenizer.decode(tokens[i]))
    return keywords[::-1]


def fill_missing_keywords(df):
    for i in range(len(df)):
        if pd.isnull(df.loc[i, 'keyword']):
                      keywords = extract_keywords(df.loc[i, 'clean_text'])
            df.loc[i, 'keyword'] = ', '.join(keywords)
    return df


def preprocess_data(train_csv_path, test_csv_path):
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    train = train_df.drop(columns=['location'])
    test = test_df.drop(columns=['location'])

    train = process_text_data(train)
    test = process_text_data(test)

    train = create_stat(train)
    test = create_stat(test)

    train = fill_missing_keywords(train)
    test = fill_missing_keywords(test)

    return train, test


if __name__ == "__main__":
    train_csv_path = "/kaggle/input/nlp-getting-started/train.csv"
    test_csv_path = "/kaggle/input/nlp-getting-started/test.csv"

    train_data, test_data = preprocess_data(train_csv_path, test_csv_path)

    print("Training data:")
    print(train_data.head())

    print("Testing data:")
    print(test_data.head())     
      
