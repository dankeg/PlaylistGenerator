import pandas as pd
import re
import nltk
from sklearn.preprocessing import MinMaxScaler
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
nltk.download('stopwords')

# Load dataset
data = pd.read_csv('../Datasets/spotify_songs.csv')

# Select relevant columns
columns_to_keep = [
    'track_id', 'track_name', 'track_artist', 'lyrics',
    'track_popularity', 'playlist_genre', 'playlist_subgenre', 'danceability',
    'energy', 'loudness', 'mode', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo', 'language'
]
data = data[columns_to_keep]

# Drop rows with missing values
data = data.dropna()

# Normalize numerical columns
numerical_columns = [
    'track_popularity', 'danceability', 'energy', 'loudness', 
    'speechiness', 'acousticness', 'instrumentalness', 
    'liveness', 'valence', 'tempo'
]
scaler = MinMaxScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])


data['lyric_length'] = data['lyrics'].apply(lambda x: len(x.split()))
data['sentence_count'] = data['lyrics'].apply(lambda x: len(re.split(r'[.!?]', x)))

# Analyze genre distribution (imbalance check)
genre_counts = data['playlist_genre'].value_counts()
print("Genre Distribution:\n", genre_counts)

# Text cleaning, stopwords removal
stop_words = set(stopwords.words('english'))
def advanced_preprocess_lyrics(lyrics):
    lyrics = re.sub(r'[^\w\s]', '', lyrics)  # Remove punctuation
    lyrics = lyrics.lower()  # Convert to lowercase
    tokens = lyrics.split()  # Tokenize
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return ' '.join(tokens)

data['cleaned_lyrics'] = data['lyrics'].apply(advanced_preprocess_lyrics)

label_encoder_genre = LabelEncoder()
label_encoder_subgenre = LabelEncoder()

data['genre_encoded'] = label_encoder_genre.fit_transform(data['playlist_genre'])
data['subgenre_encoded'] = label_encoder_subgenre.fit_transform(data['playlist_subgenre'])

# Save mappings for later interpretation
genre_mapping = dict(zip(label_encoder_genre.classes_, label_encoder_genre.transform(label_encoder_genre.classes_)))
subgenre_mapping = dict(zip(label_encoder_subgenre.classes_, label_encoder_subgenre.transform(label_encoder_subgenre.classes_)))

print("Genre Mapping:", genre_mapping)
print("Subgenre Mapping:", subgenre_mapping)

# Save the cleaned data
data.to_csv('../Datasets/cleaned_spotify_songs.csv', index=False)

# List of words to exclude from the WordCloud
exclude_list = []


def remove_excluded_words(lyrics, exclude_list):
    words = lyrics.split()
    filtered = [word for word in words if word not in exclude_list]
    return ' '.join(filtered)

# Combine all lyrics, excluding words from the exclude list
filtered_lyrics = ' '.join(
    data['cleaned_lyrics'].apply(lambda x: remove_excluded_words(x, exclude_list))
)

# Generate the WordCloud with filtered lyrics
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(filtered_lyrics)

# Plot the WordCloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Most Common Words in Song Lyrics (Filtered)")
plt.show()
