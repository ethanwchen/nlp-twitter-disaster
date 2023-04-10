import re
import string
import pandas as pd
from wordcloud import STOPWORDS
import transformers
import torch


def extract_hashtags(tweet):
    return " ".join([match.group(0)[1:] for match in re.finditer(r"#\w+", tweet)]) or 'no'


def extract_mentions(tweet):
    return " ".join([match.group(0)[1:] for match in re.finditer(r"@\w+", tweet)]) or 'no'


def extract_links(tweet):
    return " ".join([match.group(0)[:] for match in re.finditer(r"https?://\S+", tweet)]) or 'no'


def process_text_data(df):
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


def fill_missing_keywords(df):
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    model = transformers.BertModel.from_pretrained('bert-base-uncased')

    def extract_keywords(text):
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

    for i in range(len(df)):
        if pd.isnull(df.loc[i, 'keyword']):
            keywords = extract_keywords(df.loc[i, 'clean_text'])
            df.loc[i, 'keyword'] = ', '.join(keywords)
    return df


def apply_features(df, clean_text_func):
    df['clean_text'] = df['text'].apply(lambda x: clean_text_func(x))
    df = process_text_data(df)
    df = create_stat(df)
    df = fill_missing_keywords(df)
    return df
