import requests

# Set up Genius API credentials
genius_token = "YOUR_GENIUS_API_TOKEN"

def search_song_on_genius(song_name, artist_name):
    """
    Searches for a song on Genius and returns the first matching result's URL.
    """
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
    """
    Scrapes the lyrics from the Genius song URL.
    """
    from bs4 import BeautifulSoup
    import requests
    
    page = requests.get(song_url)
    soup = BeautifulSoup(page.content, "html.parser")
    lyrics_div = soup.find("div", class_="lyrics")  # Genius stores lyrics here on most pages
    if lyrics_div:
        return lyrics_div.get_text()
    else:
        # For pages with different HTML structure
        lyrics = "\n".join([line.get_text() for line in soup.find_all("p")])
        return lyrics if lyrics else "Lyrics not found."

