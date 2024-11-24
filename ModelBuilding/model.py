import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('../Datasets/processed_spotify_songs_step3.csv')

# Vectorize lyrics using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['lyrics'].fillna(''))

# Compute similarity matrix
lyrics_similarity = cosine_similarity(tfidf_matrix)

# Recommend similar tracks for a given track_id
track_indices = {track_id: idx for idx, track_id in enumerate(data['id'])}
def recommend_similar_tracks(track_id, similarity_matrix, top_n=20):
    idx = track_indices[track_id]
    similar_indices = similarity_matrix[idx].argsort()[-top_n-1:-1][::-1]
    return data.iloc[similar_indices][['name', 'artists']]

index = 9468

# Example recommendation for a specific track
recommendations = recommend_similar_tracks(data['id'].iloc[index], lyrics_similarity)
print('Recommendations for:', data['name'].iloc[index], 'by', data['artists'].iloc[index])
print(recommendations)

