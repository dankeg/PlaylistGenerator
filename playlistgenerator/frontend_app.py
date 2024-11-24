import streamlit as st

# App Title
st.title("🎵 Personalized Playlist Generator")

# Sidebar
st.sidebar.header("Your Preferences")

# User Spotify Account Input
spotify_account = st.sidebar.text_input("Enter your Spotify username or email:", "")

# Dropdown for mood
mood = st.sidebar.selectbox("Select Mood", ["Happy", "Relaxed", "Energetic", "Sad"])

# Slider for number of recommendations
num_recommendations = st.sidebar.slider("Number of Recommendations", 1, 20, 10)

# Sidebar button
if st.sidebar.button("Generate Playlist"):
    if spotify_account:
        st.sidebar.success("Preferences Saved! Generating playlist for Spotify account: " + spotify_account)
    else:
        st.sidebar.error("Please enter your Spotify account to continue.")

# Main Content Area
st.header("Your Personalized Playlist")

# Placeholder for Playlist
st.subheader("Recommendations:")
st.write("Here will be your personalized song recommendations!")

# Add Skip and Like Buttons
col1, col2 = st.columns(2)

# Skip Button
if col1.button("⏭️ Generate Again"):
    st.write("You skipped the current list!")

# Like Button
if col2.button("❤️ Like"):
    st.write("You liked the current list!")

# Footer
st.markdown("---")
# Placeholder for Error/Info Messages
st.subheader("⚠️ Error Messages")
st.info("Placeholder for error messages.")
