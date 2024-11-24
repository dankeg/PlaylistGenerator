import pandas as pd
import numpy as np

# Load the dataset (Replace 'file_path.csv' with the actual file path)
file_path = "spotify_songs.csv"
data = pd.read_csv(file_path)

# Display the first few rows to understand the data structure
print("Dataset Preview:")
print(data.head())

# Preprocessing
# 1. Handle missing values
data = data.fillna({
    'track_name': 'Unknown',
    'track_artist': 'Unknown',
    'lyrics': 'No Lyrics',
    'track_album_name': 'Unknown',
    'track_album_release_date': 'Unknown',
    'playlist_name': 'Unknown',
    'playlist_genre': 'Unknown',
    'playlist_subgenre': 'Unknown',
    'danceability': data['danceability'].mean(),
    'energy': data['energy'].mean(),
    'key': -1,  # Use -1 for missing keys
    'loudness': data['loudness'].mean(),
    'mode': 1,  # Defaulting to major
    'speechiness': data['speechiness'].mean(),
    'acousticness': data['acousticness'].mean(),
    'instrumentalness': data['instrumentalness'].mean(),
    'liveness': data['liveness'].mean(),
    'valence': data['valence'].mean(),
    'tempo': data['tempo'].mean(),
    'duration_ms': data['duration_ms'].mean(),
    'language': 'Unknown'
})

# 2. Convert dates to a standard format
data['track_album_release_date'] = pd.to_datetime(
    data['track_album_release_date'], errors='coerce'
)

# 3. Handle categorical data (e.g., language, playlist_genre)
data['language'] = data['language'].astype('category')
data['playlist_genre'] = data['playlist_genre'].astype('category')
data['playlist_subgenre'] = data['playlist_subgenre'].astype('category')

# 4. Normalize numerical columns for analysis
numerical_columns = [
    'danceability', 'energy', 'key', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence',
    'tempo', 'duration_ms'
]

# Normalization using Min-Max Scaling
data[numerical_columns] = (data[numerical_columns] - data[numerical_columns].min()) / \
                          (data[numerical_columns].max() - data[numerical_columns].min())

# Feature Engineering (optional)
# Example: Create a new column for song length in minutes
data['duration_min'] = data['duration_ms'] / 60000
data = data[data['language'] == 'en']
# Save preprocessed data to a new CSV file
output_path = "preprocessed_data.csv"
data.to_csv(output_path, index=False)
print(f"Preprocessed dataset saved to {output_path}")

# Quick Data Overview
print("\nData Info:")
print(data.info())

print("\nSummary Statistics:")
print(data.describe())





# Get unique values for a specific column
specific_value_count = data['language'].value_counts()
print(specific_value_count)
