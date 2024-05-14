import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk import tokenize
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from PIL import Image

df = pd.read_excel('engelbert_songs.xlsx')

df = df[df['word_count'] > 1]

w_tokenizer = tokenize.WhitespaceTokenizer()

# tool that transforms tokens into lemmas. Lemma is the base or dictionary form of a word
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text.lower()) if w not in stopwords.words('english')]

df['Lemmas'] = df['lyrics'].apply(lambda x : lemmatize_text(x))

# WC for unigrams
# Convert a collection of text documents to a matrix of token counts:
# The lower and upper boundary of the range for different word n-grams
# (1,1) means unigrams, (2, 2) means bigrams
vectorizer = CountVectorizer(ngram_range=(1,1))
bag_of_words = vectorizer.fit_transform(df['Lemmas'].apply(lambda x : ' '.join(x)))
sum_words = bag_of_words.sum(axis=0) 
# frequencies of words:
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
# sorted:
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
words_dict = dict(words_freq)
WC_height = 1000
WC_width = 1500
WC_max_words = 30
wordCloud = WordCloud(max_words = WC_max_words, height = WC_height, width = WC_width)

wordCloud.generate_from_frequencies(words_dict)
plt.figure(figsize=(10,8))
plt.imshow(wordCloud)
plt.title('Word Cloud', size = 25)
plt.axis("off")
plt.savefig('assets/visual_1.png')
plt.show()

# WC for bigrams
vectorizer = CountVectorizer(ngram_range=(2,2))
bag_of_words = vectorizer.fit_transform(df['Lemmas'].apply(lambda x : ' '.join(x)))
sum_words = bag_of_words.sum(axis=0) 
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
words_dict = dict(words_freq)
WC_height = 1000
WC_width = 1500
WC_max_words = 30
wordCloud = WordCloud(max_words = WC_max_words, height = WC_height, width = WC_width, colormap='gist_gray')

wordCloud.generate_from_frequencies(words_dict)
plt.figure(figsize=(10,8))
plt.imshow(wordCloud)
plt.title('Word Cloud of bigrams', fontsize = 25)
plt.axis("off")
plt.savefig('assets/visual_2.png')
plt.show()

# sentiment analysis:
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# adding new columns
df['lyrics_negative'] = ''
df['lyrics_neutral'] = ''
df['lyrics_positive'] = ''
df['lyrics_compound'] = ''

for index, row in df.iterrows():
    lyrics = row['lyrics']
    # calculating scores for song: 
    lyrics_sentiments = sia.polarity_scores(lyrics)
    lyrics_negative, lyrics_neutral, lyrics_positive, lyrics_compound = lyrics_sentiments.values()
    # Filling in sentiment columns:
    df.at[index, 'lyrics_negative'] = lyrics_negative
    df.at[index, 'lyrics_neutral'] = lyrics_neutral
    df.at[index, 'lyrics_positive'] = lyrics_positive
    df.at[index, 'lyrics_compound'] = lyrics_compound
    
def interpret_sentiment(score):
    if score > 0.2:
        return 'positive'
    elif score < -0.2:
        return 'negative'
    else:
        return 'neutral'
    
df['sentiment'] = df['lyrics_compound'].apply(interpret_sentiment)


# Count the number of items for each sentiment category
sentiment_counts = df['sentiment'].value_counts()

# Define custom colors based on sentiment
colors = {'positive': 'green', 'negative': 'red', 'neutral': 'skyblue'}

# Create pie chart
plt.figure(figsize=(10, 8))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, colors=[colors.get(x, 'gray') for x in sentiment_counts.index], textprops={'fontsize': 18})
plt.title('% of songs per sentiment', size = 18)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.savefig('assets/visual_3.png')
plt.show()

# Histogram of coumpound score
plt.figure(figsize=(10, 8))
plt.hist(df['lyrics_compound'], bins=10, color='navy', edgecolor='black')
plt.title('Compound Score Distribution', fontsize=18)
plt.xlabel('Compound Score', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('assets/visual_4.png')

# Histogram of positive score
plt.figure(figsize=(10, 8))
plt.hist(df['lyrics_positive'], bins=10, color='navy', edgecolor='black')
plt.title('Positive Score Distribution', fontsize=18)
plt.xlabel('Positive score', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('assets/visual_5.png')

# Histogram of negative score
plt.figure(figsize=(10, 8))
plt.hist(df['lyrics_negative'], bins=10, color='navy', edgecolor='black')
plt.title('Negative Score Distribution', fontsize=18)
plt.xlabel('Negative score', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('assets/visual_6.png')

# Histogram of word count
plt.figure(figsize=(10, 8))
plt.hist(df['word_count'], bins=10, color='navy', edgecolor='black')
plt.title('Word Count Distribution', fontsize=18)
plt.xlabel('Word count', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('assets/visual_7.png')

top_counts = df.sort_values(by='word_count', ascending=False).head(3)
# Plotting
plt.figure(figsize=(10, 8))
plt.bar(top_counts['title'], top_counts['word_count'], color='navy')
plt.title('Top 3 Songs with Highest Word Count', fontsize=18)
plt.xlabel('Song Title')
plt.ylabel('Word Count')
plt.xticks(fontsize=12)
plt.yticks(fontsize=16)
plt.xticks(rotation=20)
plt.savefig('assets/visual_8.png', bbox_inches='tight')
plt.show()

lowest_counts = df.sort_values(by='word_count', ascending=True).head(3)
plt.figure(figsize=(10, 8))
plt.bar(lowest_counts['title'], lowest_counts['word_count'], color='navy')
plt.title('Top 3 Songs with Lowest Word Count', fontsize=18)
plt.xlabel('Song Title')
plt.ylabel('Word Count')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xticks(rotation=20)
plt.savefig('assets/visual_9.png', bbox_inches='tight')
plt.show()

concatenated_string = ' '.join([word for sublist in df['Lemmas'] for word in sublist])
mask = np.array(Image.open("assets/engelbert.png"))  # Provide the path to your desired shape image
wordcloud = WordCloud(width=800, height=400, background_color='white', max_words = 10000, colormap='gist_gray', mask=mask).generate(concatenated_string)

# Plot the word cloud
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig('assets/visual_10.png')
plt.show()






