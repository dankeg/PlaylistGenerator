from textblob import TextBlob
import gensim
from gensim import corpora
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from data import *
from Preprocess import *

def perform_sentiment_analysis(df):
    """Performs sentiment analysis on lyrics."""
    def get_sentiment(text):
        if not text:
            return 0
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    df['sentiment'] = df['cleaned_lyrics'].apply(get_sentiment)
    return df

def perform_topic_modeling(df):
    """Performs topic modeling on lyrics."""
    # Create a dictionary and corpus
    dictionary = corpora.Dictionary(df['tokenized_lyrics'])
    corpus = [dictionary.doc2bow(text) for text in df['tokenized_lyrics']]

    # Build LDA model
    lda_model = gensim.models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)

    # Assign topics to documents
    def get_dominant_topic(bow):
        topics = lda_model.get_document_topics(bow)
        topics = sorted(topics, key=lambda x: x[1], reverse=True)
        return topics[0][0] if topics else None

    df['dominant_topic'] = [get_dominant_topic(bow) for bow in corpus]
    return df, lda_model

# Perform NLP analysis
df = perform_sentiment_analysis(df)
df, lda_model = perform_topic_modeling(df)
print(df[['track_name', 'sentiment', 'dominant_topic']].head())


def vectorize_lyrics(df):
    """Vectorizes lyrics using TF-IDF."""
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['cleaned_lyrics'])
    return tfidf_matrix

def compute_similarity(df, tfidf_matrix):
    """Computes similarity based on audio features and lyrics."""
    # Numerical features
    numerical_features = ['danceability', 'energy', 'loudness', 'speechiness',
                          'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    audio_features_matrix = df[numerical_features].values

    # Normalize audio features
    audio_features_matrix = MinMaxScaler().fit_transform(audio_features_matrix)

    # Compute similarities
    audio_similarity = cosine_similarity(audio_features_matrix)
    lyrics_similarity = cosine_similarity(tfidf_matrix)

    # Combine similarities
    combined_similarity = (audio_similarity + lyrics_similarity) / 2
    return combined_similarity

def get_recommendations(track_id, df, similarity_matrix, top_n=5):
    """Recommends tracks similar to the given track ID."""
    indices = pd.Series(df.index, index=df['track_id']).drop_duplicates()
    idx = indices[track_id]

    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:]  # Exclude the track itself

    track_indices = [i[0] for i in sim_scores]
    final_df = df.iloc[track_indices][['track_name', 'artist_name', 'sentiment', 'dominant_topic']]
    return final_df.drop_duplicates(subset="track_name").head(top_n)

# Vectorize lyrics
tfidf_matrix = vectorize_lyrics(df)

# Compute similarity matrix
similarity_matrix = compute_similarity(df, tfidf_matrix)

# Get recommendations
sample_track_id = df['track_id'].iloc[0]
recommendations = get_recommendations(sample_track_id, df, similarity_matrix, top_n=5)
print("Recommendations:")
print(recommendations)

