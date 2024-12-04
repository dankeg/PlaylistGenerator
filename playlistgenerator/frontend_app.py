import random

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tf_agents.trajectories import time_step as ts

from playlistgenerator.ReinforcementLearning.utils import small_hash, softmax


class RLUpdate:
    """Handles interaction with the RL mode, and provides updates to the UI"""

    def __init__(self):
        # Load data
        self.saved_data = pd.read_csv(
            "playlistgenerator/Datasets/combined_recommendation.csv"
        )

        # Copy the data and add unique_id and user_score columns
        self.saved_data["unique_id"] = range(1, len(self.saved_data) + 1)
        self.processed_data = self.saved_data.copy()
        self.processed_data["user_score"] = 0

        # Hash track_name and artist_name for unique identification
        self.processed_data["name"] = self.processed_data["name"].apply(
            lambda x: small_hash(x)
        )
        self.processed_data["artists"] = self.processed_data["artists"].apply(
            lambda x: small_hash(x)
        )

        # Initialize the state with the first 5 rows of data
        self.state = self.processed_data.head(5).copy()
        self.episode_ended = False

        # Load the saved policy
        self.saved_policy = tf.saved_model.load(
            "playlistgenerator/outputs/ModelOutputs/policy"
        )

        # Initialize variables
        self.bar_chart = []

    def fetch_playlist(self, num: int) -> list[str]:
        """Fetch the playlist, once a sufficient number of songs have been liked.

        Args:
            num (int): Number of songs to include in the playlist.

        Returns:
            list[str]: Generated playlist.
        """
        output_list = []

        for x in range(num):
            id_value = self.processed_data.iloc[x]["unique_id"]

            # Find the row in the 'saved_data' DataFrame where the 'unique_id' matches
            matching_row = self.saved_data[self.saved_data["unique_id"] == id_value]
            track_name = matching_row["name"].values[0]
            artist_name = matching_row["artists"].values[0]

            output_list.append(track_name + " - " + artist_name)

        return output_list

    def run_model(self, opinion: str) -> str:
        """Execute the mode, given the response of the user.

        Args:
            opinion (str): User response to the current selection.

        Returns:
            str: Next song for a user to like or dislike.
        """
        # Prepare the observation tensor
        observation_tensor = self.state.to_numpy().flatten()
        observation_tensor = tf.expand_dims(
            observation_tensor, axis=0
        )  # Add batch dimension

        # Create the time_step object
        step_type = tf.constant([1], dtype=tf.int32)  # MID step type
        reward = tf.constant([0.0], dtype=tf.float32)  # Reward from the environment
        discount = tf.constant(
            [0.99], dtype=tf.float32
        )  # Discount factor for this step
        time_step = ts.TimeStep(
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=observation_tensor,
        )

        # Call the 'action' signature explicitly
        action_output = self.saved_policy.action(time_step=time_step)

        # Extract and print the resulting action
        action_value = action_output[0].numpy()[0]
        id_value = self.state.iloc[action_value]["unique_id"]

        # Find the row in the 'saved_data' DataFrame where the 'unique_id' matches
        matching_row = self.saved_data[self.saved_data["unique_id"] == id_value]
        track_name = matching_row["name"].values[0]
        artist_name = matching_row["artists"].values[0]

        # Print the track and artist names
        print("\n")
        print(track_name)
        print(artist_name)
        print("\n")

        # Append the track name to the bar_chart
        self.bar_chart.append(track_name)

        # Calculate softmax probabilities
        score_array = self.state["user_score"].to_numpy()
        softmax_output = softmax(score_array)

        # Determine if the action is selected based on the softmax probability
        probability = softmax_output[action_value]
        selection = random.random() < probability

        if opinion != "dislike":
            # If the action is selected, update the user score and reorder the data
            id = id_value
            self.state.loc[self.state["unique_id"] == id, "user_score"] += 1
            self.processed_data.loc[
                self.processed_data["unique_id"] == id, "user_score"
            ] += 1

            # Swap rows and move the selected row to the end
            self.processed_data.iloc[[4, 5]] = (
                self.processed_data.iloc[[5, 4]].copy().values
            )
            row_to_move = self.processed_data.iloc[5:6].copy()
            self.processed_data = pd.concat(
                [self.processed_data.drop(self.processed_data.index[5]), row_to_move],
                ignore_index=True,
            )

            # Update the state with the first 5 rows of data
            self.state = self.processed_data.iloc[:5].copy()
        else:
            # If the action is not selected, update the user score and reorder the data
            id = id_value
            self.state.loc[self.state["unique_id"] == id, "user_score"] += -1
            self.processed_data.loc[
                self.processed_data["unique_id"] == id, "user_score"
            ] += -1

            # Swap rows and move the selected row to the end
            self.processed_data.iloc[[action_value, 5]] = (
                self.processed_data.iloc[[5, action_value]].copy().values
            )
            row_to_move = self.processed_data.iloc[5:6].copy()
            self.processed_data = pd.concat(
                [self.processed_data.drop(self.processed_data.index[5]), row_to_move],
                ignore_index=True,
            )

            # Update the state with the first 5 rows of data
            self.state = self.processed_data.iloc[:5].copy()

        return track_name + " - " + artist_name


# Initialize RLUpdate class at runtime (only once)
if "RLUpdate_instance" not in st.session_state:
    st.session_state.RLUpdate_instance = RLUpdate()

if "like_count" not in st.session_state:
    st.session_state.like_count = 0

if "prev_result" not in st.session_state:
    st.session_state.prev_result = ""


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
        st.sidebar.success(
            "Preferences Saved! Generating playlist for Spotify account: "
            + spotify_account
        )
    else:
        st.sidebar.error("Please enter your Spotify account to continue.")

# Main Content Area
st.header("Your Personalized Playlist")
playlist_placeholder = st.container()

# Placeholder for Playlist
st.subheader("Recommendations:")
st.write("Here will be your personalized song recommendations!")

# Placeholder for function output
result_placeholder = st.empty()

# Add Dislike and Like Buttons
col1, col2, col3 = st.columns(3)

# Dislike Button
if col1.button("💔 Dislike"):
    result = st.session_state.RLUpdate_instance.run_model("dislike")

    while result == st.session_state.prev_result:
        result = st.session_state.RLUpdate_instance.run_model("like")

    st.session_state.prev_result = result

    result_placeholder.write(result)

# Like Button
if col2.button("❤️ Like"):
    result = st.session_state.RLUpdate_instance.run_model("like")

    while result == st.session_state.prev_result:
        result = st.session_state.RLUpdate_instance.run_model("dislike")

    st.session_state.prev_result = result

    result_placeholder.write(result)
    st.session_state.like_count += 1

    if st.session_state.like_count > num_recommendations:
        playlist = st.session_state.RLUpdate_instance.fetch_playlist(
            num_recommendations
        )
        for song in playlist:
            with playlist_placeholder:
                st.write(song)

# Like Button
if col3.button("Start!"):
    result = st.session_state.RLUpdate_instance.run_model("like")
    result_placeholder.write(result)

# Footer
st.markdown("---")
# Placeholder for Error/Info Messages
st.subheader("⚠️ Error Messages")
st.info("Placeholder for error messages.")
