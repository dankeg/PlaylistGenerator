import numpy as np
import pandas as pd
import re
import string
from sklearn.preprocessing import MinMaxScaler
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from data import *

# Download NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_data(df):
    """Preprocesses the collected data."""
    # Normalize numerical audio features
    numerical_features = ['danceability', 'energy', 'loudness', 'speechiness',
                          'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    scaler = MinMaxScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # Clean lyrics
    def clean_lyrics(text):
        if pd.isna(text):
            return ''
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = text.strip()
        return text

    df['cleaned_lyrics'] = df['lyrics'].apply(clean_lyrics)

    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    df['tokenized_lyrics'] = df['cleaned_lyrics'].apply(
        lambda x: [word for word in word_tokenize(x) if word not in stop_words]
    )

    # Encode 'played_at' as a timestamp
    # df['played_at_timestamp'] = pd.to_datetime(df['played_at']).astype(np.int64) // 10**9

    return df

# Preprocess data
df = preprocess_data(df)
print(df[['track_name', 'cleaned_lyrics', 'tokenized_lyrics']].head())
