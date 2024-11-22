import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Spotify API credentials
client_id = '2cb38f81b5094e57a4b67ec9d27d20c4'
client_secret = '35625fcd626b4566ae712a75aa14fd81'
redirect_uri = 'http://127.0.0.1:9090/'

# Genius API token
genius_token = 'LYDItbbdt-WqO1eWl05P-cIRABGIWTYCTfGcYT4K4IETES_1lnUlmluZR3DUbYWX'

# Scopes for Spotify API
scope = "user-read-private user-follow-read user-top-read playlist-read-private user-read-recently-played"

# Initialize Spotify client
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                               client_secret=client_secret,
                                               redirect_uri=redirect_uri,
                                               scope=scope))

def get_recent_tracks():
    """Fetches the user's recently played tracks."""
    results = sp.current_user_recently_played(limit=50)
    tracks = []
    for item in results['items']:
        track = item['track']
        track_info = {
            'track_id': track['id'],
            'track_name': track['name'],
            'artist_name': ', '.join([artist['name'] for artist in track['artists']]),
            'played_at': item['played_at']
        }
        tracks.append(track_info)
    return tracks

def get_audio_features(track_ids):
    """Fetches audio features for a list of track IDs."""
    features = sp.audio_features(track_ids)
    return features

def search_song_on_genius(song_name, artist_name):
    """Searches for a song on Genius and returns the song URL."""
    base_url = "https://api.genius.com"
    headers = {"Authorization": f"Bearer {genius_token}"}
    search_url = f"{base_url}/search"
    params = {'q': f"{song_name} {artist_name}"}
    
    response = requests.get(search_url, headers=headers, params=params)
    if response.status_code == 200:
        json_data = response.json()
        song_info = None
        for hit in json_data["response"]["hits"]:
            if artist_name.lower() in hit["result"]["primary_artist"]["name"].lower():
                song_info = hit
                break
        if song_info:
            return song_info["result"]["url"]
    return None

def get_lyrics(song_url):
    """Scrapes lyrics from the Genius song URL."""
    page = requests.get(song_url)
    soup = BeautifulSoup(page.content, "html.parser")
    lyrics_div = soup.find("div", class_="lyrics")
    if lyrics_div:
        return lyrics_div.get_text()
    else:
        # For pages with different HTML structure
        lyrics = "\n".join([line.get_text() for line in soup.find_all("p")])
        return lyrics if lyrics else None

def collect_user_data():
    """Collects user listening history, audio features, and lyrics."""
    tracks = get_recent_tracks()
    track_ids = [track['track_id'] for track in tracks]
    audio_features = get_audio_features(track_ids)
    
    data = []
    for i, track in enumerate(tracks):
        track_data = track.copy()
        # Merge audio features
        if audio_features[i]:
            track_data.update(audio_features[i])
        else:
            continue  # Skip if no audio features are found
        
        # Get lyrics
        song_url = search_song_on_genius(track['track_name'], track['artist_name'])
        if song_url:
            lyrics = get_lyrics(song_url)
            track_data['lyrics'] = lyrics
        else:
            track_data['lyrics'] = None
        
        data.append(track_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    # Save to CSV
    df.to_csv('user_data.csv', index=False)
    return df

# Collect data
df = collect_user_data()
print(df.head())
