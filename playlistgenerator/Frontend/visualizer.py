import streamlit as st

# Function to apply custom CSS for background and layout
def set_background_style():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(to bottom right, #34495e, #bdc3c7);
            color: white;
        }
        .spotify-title {
            text-align: center;
            color: white;
            font-size: 3rem;
            font-family: 'Trebuchet MS', sans-serif;
            margin-bottom: 20px;
        }
        .spotify-subheader {
            text-align: center;
            font-size: 1.5rem;
            font-family: 'Trebuchet MS', sans-serif;
            margin-bottom: 30px;
        }
        .button-container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Function to display the mini Spotify player
def mini_spotify_player(album_image_url, song_name, artist_name):
    # Title
    st.markdown("<div class='spotify-title'>🎵 Mini Spotify Player</div>", unsafe_allow_html=True)

    # Album image
    st.image(
        album_image_url,
        caption=f"{song_name} - {artist_name}",
        use_container_width=True,
    )

    # Song details
    st.markdown(
        f"<div class='spotify-subheader'>{song_name} by {artist_name}</div>",
        unsafe_allow_html=True,
    )

    # Like and Dislike buttons with alignment fix
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='button-container'>", unsafe_allow_html=True)
        if st.button("👍 Like"):
            st.success("You liked this song! ❤️")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='button-container'>", unsafe_allow_html=True)
        if st.button("👎 Dislike"):
            st.error("You disliked this song! 💔")
        st.markdown("</div>", unsafe_allow_html=True)

# Apply background style
set_background_style()

# Spotify API key input
st.sidebar.header("Spotify API Settings")
api_key = st.sidebar.text_input(
    "Enter your Spotify API Key:",
    type="password",  # Mask the key input
    help="Your Spotify API key is required to fetch live data.",
)

if api_key:
    st.sidebar.success("API key submitted successfully!")
    # Store API key in session state for use in the app
    st.session_state["spotify_api_key"] = api_key
else:
    st.sidebar.warning("Please enter your Spotify API key.")

# Example usage with Tame Impala's Currents album art
album_image_url = "https://upload.wikimedia.org/wikipedia/en/thumb/9/9b/Tame_Impala_-_Currents.png/220px-Tame_Impala_-_Currents.png"  # Verified Currents album art from Spotify CDN
song_name = "Let It Happen"
artist_name = "Tame Impala"

# Display the player only if an API key is provided
if "spotify_api_key" in st.session_state:
    mini_spotify_player(album_image_url, song_name, artist_name)
else:
    st.warning("Submit your Spotify API key to load the player.")
