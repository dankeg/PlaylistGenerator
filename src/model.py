import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from scipy.sparse.linalg import svds
from textblob import TextBlob
from wordcloud import WordCloud
import seaborn as sns

# Load the dataset
file_path = "preprocessed_data.csv"  
data = pd.read_csv(file_path)

# Preprocess the dataset
data['lyrics'] = data['lyrics'].fillna("").str.lower().str.replace(r"[^\w\s]", "", regex=True)

# NLP Analysis: Sentiment and Emotional Analysis
data['polarity'] = data['lyrics'].apply(lambda x: TextBlob(x).sentiment.polarity)
data['subjectivity'] = data['lyrics'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# Emotion Classification
def classify_emotion(polarity):
    if polarity > 0.2:
        return "Positive"
    elif polarity < -0.2:
        return "Negative"
    else:
        return "Neutral"

data['emotion'] = data['polarity'].apply(classify_emotion)

# Visualization: Word Cloud
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(data['lyrics']))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud of Lyrics")
plt.show()

# Visualization: Emotion Distribution
emotion_counts = data['emotion'].value_counts()
emotion_counts.plot(kind='bar', color=['green', 'blue', 'red'], figsize=(8, 5), title='Emotion Distribution')
plt.xlabel('Emotion')
plt.ylabel('Frequency')
plt.show()

# Topic Modeling (LDA)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
lyrics_tfidf = vectorizer.fit_transform(data['lyrics'])
lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_model.fit(lyrics_tfidf)

print("\nTop Words in Each Topic:")
for idx, topic in enumerate(lda_model.components_):
    print(f"Topic {idx+1}: ", [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])

# Cosine Similarity
cosine_sim_matrix = cosine_similarity(lyrics_tfidf)
print("\nCosine Similarity Matrix (First 5 Songs):")
print(cosine_sim_matrix[:5, :5])  # Display a sample of similarity scores

# Machine Learning Models
# Prepare features and target variable
data['popularity_label'] = (data['track_popularity'] > 50).astype(int)
X_train, X_test, y_train, y_test = train_test_split(lyrics_tfidf, data['popularity_label'], test_size=0.2, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
}

# Train and evaluate models
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    results[model_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
    }
    print(f"\n{model_name} Classification Report:\n")
    print(classification_report(y_test, y_pred))

# Collaborative Filtering (Matrix Factorization)
user_song_matrix = pd.pivot_table(data, index='playlist_id', columns='track_id', values='track_popularity').fillna(0)
matrix = user_song_matrix.to_numpy()

# SVD for Matrix Factorization
u, sigma, vt = svds(matrix, k=50)  # Reduce dimensions
sigma = np.diag(sigma)

# Reconstruct ratings
predicted_ratings = np.dot(np.dot(u, sigma), vt)
predicted_ratings_df = pd.DataFrame(predicted_ratings, index=user_song_matrix.index, columns=user_song_matrix.columns)

print("\nSample Predicted Ratings:")
print(predicted_ratings_df.head())

# Hybrid Recommendation: Combine Collaborative Filtering with Cosine Similarity
song_idx = 10  
similar_songs = cosine_sim_matrix[song_idx].argsort()[-6:-1][::-1]  
print(f"\nTop 5 Songs Similar to '{data.iloc[song_idx]['track_name']}':")
for idx in similar_songs:
    print(data.iloc[idx]['track_name'])

# Identify the Best Model
results_df = pd.DataFrame(results).T
best_model = results_df['Accuracy'].idxmax()
print(f"\nBest Model: {best_model}\n")

# Visualize Model Performance
results_df.plot(kind="bar", figsize=(10, 6), title="Model Performance Metrics", ylim=(0, 1))
plt.ylabel("Score")
plt.show()
