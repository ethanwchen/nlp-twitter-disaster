# Import necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS
from nltk import FreqDist, word_tokenize

def plot_countplot(train):
    sns.set_palette(['#a2d5c6', '#f9989f'])
    sns.countplot(x='target', data=train)
    plt.title('Distribution')
    print(train['target'].value_counts())

def plot_bar_chart(train):
    all_text = ' '.join(train['clean_text'].tolist())
    words = all_text.split()
    word_counts = collections.Counter(words)
    top_words = word_counts.most_common(20)

    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar([word[0] for word in top_words], [word[1] for word in top_words], color='#9dc3e6')
    ax.set_xticklabels([word[0] for word in top_words], rotation=90)
    ax.set_title('20 Most Frequent Words')
    ax.set_xlabel('Word')
    ax.set_ylabel('Frequency')
    plt.show()

def plot_top_keywords(train):
    plt.figure(figsize=(10,8))
    sns.countplot(y=train.keyword, order=train.keyword.value_counts().iloc[:20].index)
    plt.title('Top 20 keywords')
    plt.show()

def plot_word_cloud(train):
    all_text = ' '.join(train['clean_text'].tolist())
    wordcloud = WordCloud(background_color='white').generate(all_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def plot_histogram(train):
    tweet_lengths = [len(tweet.split()) for tweet in train['clean_text']]
    fig, ax = plt.subplots(figsize=(10,6))
    ax.hist(tweet_lengths, bins=50, color='#c7a9e6', alpha=0.7)
    ax.set_title('Distribution of Tweet Length')
    ax.set_xlabel('Number of Words')
    ax.set_ylabel('Frequency')
    plt.show()

def plot_heatmap(train):
    cols_of_interest = ['target', 'text_len', 'word_count', 'stop_word_count', 'mention_count', 'link_count', 'hashtag_count', 'punctuation_count', 'caps_count', 'caps_ratio']
    df = train[cols_of_interest]
    corr_matrix = df.corr()
    sns.set(rc={'figure.figsize':(10,8)})
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Coefficients between Additional Features and Target Variable')
    plt.show()

def plot_word_frequencies(train):
    stopwords = set(STOPWORDS)
    word_freq = FreqDist(w for w in word_tokenize(' '.join(train['clean_text']).lower()) if 
                         (w not in stopwords) & (w.isalpha()))
    df_word_freq = pd.DataFrame.from_dict(word_freq, orient='index', columns=['count'])
    top20w = df_word_freq.sort_values('count',ascending=False).head(20)

    plt.figure(figsize=(8,6))
    sns.barplot(x='count', y=top20w.index, data=top20w)
    plt.title('Top 20 words')
    plt.show()

def plot_top_words_disaster_and_nondisaster(train):
    plt.figure(figsize=(16,7))

    plt.subplot(121)

    freq_d = FreqDist(w for w in word_tokenize(' '.join(train.loc[train.target==1, 'clean_text']).lower()) if
    (w not in stopwords) & (w.isalpha()))

    df_d = pd.DataFrame.from_dict(freq_d, orient='index', columns=['count'])
    
 top20_d = df_d.sort_values('count',ascending=False).head(20)

sns.barplot(x='count', y=top20_d.index, data=top20_d, color='c')

plt.title('Top words in disaster tweets')

plt.subplot(122)

freq_nd = FreqDist(w for w in word_tokenize(' '.join(train.loc[train.target==0, 'clean_text']).lower()) if
(w not in stopwords) & (w.isalpha()))

df_nd = pd.DataFrame.from_dict(freq_nd, orient='index', columns=['count'])

top20_nd = df_nd.sort_values('count',ascending=False).head(20)

sns.barplot(x='count', y=top20_nd.index, data=top20_nd, color='y')

plt.title('Top words in non-disaster tweets')

plt.show()

