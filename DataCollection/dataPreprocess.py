import pandas as pd
import re
from langdetect import detect
from sklearn.preprocessing import MinMaxScaler
# Load dataset
data = pd.read_csv('../Datasets/spotify_songs.csv')

# Select relevant columns
columns_to_keep = [
    'id', 'name', 'artists', 'danceability',
    'energy', 'loudness', 'mode', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo', 'lyrics'
]

data = data[columns_to_keep]

data['artists'] = data['artists'].astype(str)

data = data.drop_duplicates()

# Drop rows with missing values
data = data.dropna()

data = data.sample(n=75000, random_state=42)

# Function to detect language of the lyrics
def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False

# Filter English songs
data = data[data['lyrics'].apply(is_english)]

# Normalize numerical columns
numerical_columns = [
    'danceability', 'energy', 'loudness', 
    'speechiness', 'acousticness', 'instrumentalness', 
    'liveness', 'valence', 'tempo'
]

scaler = MinMaxScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

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

# Text cleaning, stopword removal using custom stopwords
def advanced_preprocess_lyrics(lyrics):
    lyrics = re.sub(r'[^\w\s]', '', lyrics)  # Remove punctuation
    lyrics = lyrics.lower()  # Convert to lowercase
    tokens = lyrics.split()  # Tokenize
    tokens = [word for word in tokens if word not in custom_stopwords]  # Remove custom stopwords
    return ' '.join(tokens)

data['lyrics'] = data['lyrics'].apply(advanced_preprocess_lyrics)
data['artists'] = data['artists'].apply(advanced_preprocess_lyrics)

# Save the cleaned data
data.to_csv('../Datasets/cleaned_spotify_songs.csv', index=False)
