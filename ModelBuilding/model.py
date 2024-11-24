import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('../Datasets/processed_spotify_songs_step3.csv')

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
# Vectorize lyrics using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words=custom_stopwords_list)
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

