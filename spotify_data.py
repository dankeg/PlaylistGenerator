import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth

# Set up authentication
client_id = 'YOUR_CLIENT_ID'
client_secret = 'YOUR_CLIENT_SECRET'
redirect_uri = 'YOUR_REDIRECT_URI'

# Define the scopes required for various user data
scope = "user-read-private user-follow-read user-top-read playlist-read-private user-read-recently-played"

# Initialize Spotify API client
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, scope=scope))

def get_track_data(track_id):
    """
    Retrieves comprehensive data about a specific track, including details, audio features, and audio analysis.
    """
    try:
        # Get basic track data
        track = sp.track(track_id)
        track_data = {
            'name': track['name'],
            'album': track['album']['name'],
            'album_release_date': track['album']['release_date'],
            'artist': [artist['name'] for artist in track['artists']],
            'track_popularity': track['popularity'],
            'track_duration_ms': track['duration_ms'],
            'explicit': track['explicit'],
            'track_number': track['track_number'],
            'disc_number': track['disc_number'],
            'available_markets': track['available_markets'],
            'external_urls': track['external_urls'],
            'album_cover': track['album']['images'][0]['url'] if track['album']['images'] else None
        }

        # Get audio features
        audio_features = sp.audio_features([track_id])[0]  # Returns a list, we want the first item
        if audio_features:
            track_data.update({
                'danceability': audio_features['danceability'],
                'energy': audio_features['energy'],
                'key': audio_features['key'],
                'loudness': audio_features['loudness'],
                'mode': audio_features['mode'],
                'speechiness': audio_features['speechiness'],
                'acousticness': audio_features['acousticness'],
                'instrumentalness': audio_features['instrumentalness'],
                'liveness': audio_features['liveness'],
                'valence': audio_features['valence'],
                'tempo': audio_features['tempo'],
                'time_signature': audio_features['time_signature']
            })

        # Get audio analysis
        audio_analysis = sp.audio_analysis(track_id)
        if audio_analysis:
            track_data['audio_analysis'] = {
                'bars': audio_analysis['bars'],       # List of bars with timestamps
                'beats': audio_analysis['beats'],     # List of beats with timestamps
                'sections': audio_analysis['sections'], # List of sections with detailed info
                'segments': audio_analysis['segments'], # List of segments with detailed info
                'tatums': audio_analysis['tatums']    # List of tatums with timestamps
            }
        
        return track_data

    except Exception as e:
        print(f"Error fetching track data: {e}")
        return None

def get_artist_data(artist_id):
    """
    Retrieves comprehensive data about a specific artist, including profile details, top tracks, albums, and related artists.
    """
    try:
        # Get basic artist profile data
        artist = sp.artist(artist_id)
        artist_data = {
            'name': artist['name'],
            'genres': artist['genres'],
            'followers': artist['followers']['total'],
            'popularity': artist['popularity'],
            'profile_image': artist['images'][0]['url'] if artist['images'] else None,
            'external_url': artist['external_urls']['spotify']
        }

        # Get top tracks for the artist
        top_tracks = sp.artist_top_tracks(artist_id, country='US')
        artist_data['top_tracks'] = [
            {
                'name': track['name'],
                'album': track['album']['name'],
                'popularity': track['popularity'],
                'preview_url': track['preview_url'],
                'external_url': track['external_urls']['spotify'],
                'track_id': track['id']
            }
            for track in top_tracks['tracks']
        ]

        # Get albums by the artist
        albums = sp.artist_albums(artist_id, album_type='album', country='US', limit=50)
        artist_data['albums'] = [
            {
                'name': album['name'],
                'release_date': album['release_date'],
                'total_tracks': album['total_tracks'],
                'album_cover': album['images'][0]['url'] if album['images'] else None,
                'album_id': album['id'],
                'external_url': album['external_urls']['spotify']
            }
            for album in albums['items']
        ]

        # Get related artists
        related_artists = sp.related_artists(artist_id)
        artist_data['related_artists'] = [
            {
                'name': related_artist['name'],
                'genres': related_artist['genres'],
                'popularity': related_artist['popularity'],
                'followers': related_artist['followers']['total'],
                'profile_image': related_artist['images'][0]['url'] if related_artist['images'] else None,
                'external_url': related_artist['external_urls']['spotify']
            }
            for related_artist in related_artists['artists']
        ]
        
        return artist_data

    except Exception as e:
        print(f"Error fetching artist data: {e}")
        return None

def get_user_profile(user_id):
    """
    Retrieves basic profile data for a specific user.
    """
    try:
        user_profile = sp.user(user_id)
        profile_data = {
            'display_name': user_profile.get('display_name'),
            'followers': user_profile['followers']['total'],
            'country': user_profile.get('country'),
            'email': user_profile.get('email'),  # Only available for the authenticated user
            'profile_image': user_profile['images'][0]['url'] if user_profile['images'] else None,
            'external_url': user_profile['external_urls']['spotify']
        }
        return profile_data
    except Exception as e:
        print(f"Error fetching user profile: {e}")
        return None

def get_user_followed_artists():
    """
    Retrieves the artists followed by the authenticated user.
    """
    try:
        followed_artists = sp.current_user_followed_artists(limit=50)
        return [
            {
                'name': artist['name'],
                'genres': artist['genres'],
                'popularity': artist['popularity'],
                'followers': artist['followers']['total'],
                'profile_image': artist['images'][0]['url'] if artist['images'] else None,
                'external_url': artist['external_urls']['spotify']
            }
            for artist in followed_artists['artists']['items']
        ]
    except Exception as e:
        print(f"Error fetching followed artists: {e}")
        return None

def get_user_top_tracks():
    """
    Retrieves the top tracks for the authenticated user.
    """
    try:
        top_tracks = sp.current_user_top_tracks(limit=20, time_range='medium_term')
        return [
            {
                'name': track['name'],
                'artist': [artist['name'] for artist in track['artists']],
                'album': track['album']['name'],
                'popularity': track['popularity'],
                'track_id': track['id'],
                'preview_url': track['preview_url'],
                'external_url': track['external_urls']['spotify']
            }
            for track in top_tracks['items']
        ]
    except Exception as e:
        print(f"Error fetching top tracks: {e}")
        return None

def get_user_playlists():
    """
    Retrieves the playlists of the authenticated user.
    """
    try:
        playlists = sp.current_user_playlists(limit=20)
        return [
            {
                'name': playlist['name'],
                'total_tracks': playlist['tracks']['total'],
                'owner': playlist['owner']['display_name'],
                'public': playlist['public'],
                'collaborative': playlist['collaborative'],
                'external_url': playlist['external_urls']['spotify']
            }
            for playlist in playlists['items']
        ]
    except Exception as e:
        print(f"Error fetching user playlists: {e}")
        return None

def get_user_recently_played():
    """
    Retrieves the recently played tracks for the authenticated user.
    """
    try:
        recently_played = sp.current_user_recently_played(limit=20)
        return [
            {
                'track_name': item['track']['name'],
                'artist': [artist['name'] for artist in item['track']['artists']],
                'album': item['track']['album']['name'],
                'played_at': item['played_at'],
                'track_id': item['track']['id'],
                'external_url': item['track']['external_urls']['spotify']
            }
            for item in recently_played['items']
        ]
    except Exception as e:
        print(f"Error fetching recently played tracks: {e}")
        return None

