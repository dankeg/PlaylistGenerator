from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the cleaned dataset
data = pd.read_csv('../Datasets/cleaned_spotify_songs.csv')

# Ensure the cleaned_lyrics column has no NaN values or non-string types
data['lyrics'] = data['lyrics'].fillna('').astype(str)

# Sentiment Analysis
def analyze_sentiment(text):
    if not text.strip():  # Check if text is empty
        return 0.0  # Neutral sentiment for empty text
    blob = TextBlob(text)
    return blob.sentiment.polarity  # Polarity: -1 (negative) to 1 (positive)

data['sentiment_score'] = data['lyrics'].apply(analyze_sentiment)

# Emotional Analysis 
def categorize_emotion(sentiment_score):
    if sentiment_score > 0.2:
        return 'Positive'
    elif sentiment_score < -0.2:
        return 'Negative'
    else:
        return 'Neutral'

data['emotion'] = data['sentiment_score'].apply(categorize_emotion)

custom_stopwords = set([
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", 
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", 
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", 
    "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", 
    "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", 
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", 
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", 
    "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", 
    "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", 
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", 
    "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", 
    "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", 
    "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", 
    "yourselves", "oh", "yeah", "nah", "woah", "hey", "ah", "uh", "la", "ooh", "doo", "ba", "da", "hmm", "whoa", 
    "sha", "ye", "na-na", "ha", "dee", "yo", "ay", "baby", "girl", "boy", "hey", "hi", "bye", "ha", "ha-ha", "na-na-na", 
    "sha-la-la", "hmm-mm", "ah-ah", "oh-oh", "da-da", "uh-oh", "yeah-yeah", "do-do", "woo", "y'all", "ain't", 
    "gotta", "wanna", "gonna", "lemme", "yeah-yeah-yeah", "uh-huh", "uh-uh", "woah-woah"
])

custom_stopwords_list = list(custom_stopwords)
# Vectorize the lyrics using custom stopwords
vectorizer = CountVectorizer(max_features=1000, stop_words=custom_stopwords_list)
lyrics_matrix = vectorizer.fit_transform(data['lyrics'])

# LDA Model
n_topics = 10
lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda_model.fit(lyrics_matrix)

# Display the top words per topic
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

display_topics(lda_model, vectorizer.get_feature_names_out(), 10)

# Add dominant topic to each song
def get_dominant_topic(lda_model, lyrics_matrix):
    topic_assignments = lda_model.transform(lyrics_matrix)
    return topic_assignments.argmax(axis=1)

data['dominant_topic'] = get_dominant_topic(lda_model, lyrics_matrix)

# Sentiment Score Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data['sentiment_score'], bins=30, kde=True, color='blue')
plt.title('Distribution of Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()

# Emotion Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='emotion', data=data, palette='pastel')
plt.title('Emotion Counts')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.show()

# Topic Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='dominant_topic', data=data, palette='muted')
plt.title('Distribution of Dominant Topics')
plt.xlabel('Topic')
plt.ylabel('Count')
plt.show()

# Sentiment Scores by Topic
plt.figure(figsize=(10, 6))
sns.boxplot(x='dominant_topic', y='sentiment_score', data=data, palette='coolwarm')
plt.title('Sentiment Score Distribution Across Topics')
plt.xlabel('Dominant Topic')
plt.ylabel('Sentiment Score')
plt.show()

# Save the data 
data.to_csv('../Datasets/processed_spotify_songs_step3.csv', index=False)


