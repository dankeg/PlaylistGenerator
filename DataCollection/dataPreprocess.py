import pandas as pd
import re
from langdetect import detect
import nltk
from sklearn.preprocessing import MinMaxScaler
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud
nltk.download('stopwords')

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

# Text cleaning, stopwords removal
stop_words = set(stopwords.words('english'))
def advanced_preprocess_lyrics(lyrics):
    lyrics = re.sub(r'[^\w\s]', '', lyrics)  # Remove punctuation
    lyrics = lyrics.lower()  # Convert to lowercase
    tokens = lyrics.split()  # Tokenize
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return ' '.join(tokens)

data['lyrics'] = data['lyrics'].apply(advanced_preprocess_lyrics)
data['artists'] = data['artists'].apply(advanced_preprocess_lyrics)

# Save the cleaned data
data.to_csv('../Datasets/cleaned_spotify_songs.csv', index=False)

# List of words to exclude from the WordCloud
exclude_list = ["not a good word list to show"]

def remove_excluded_words(lyrics, exclude_list):
    words = lyrics.split()
    filtered = [word for word in words if word not in exclude_list]
    return ' '.join(filtered)

# Combine all lyrics, excluding words from the exclude list
filtered_lyrics = ' '.join(
    data['lyrics'].apply(lambda x: remove_excluded_words(x, exclude_list))
)

# Generate the WordCloud with filtered lyrics
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(filtered_lyrics)

# Plot the WordCloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Most Common Words in Song Lyrics (Filtered)")
plt.show()
