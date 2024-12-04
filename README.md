# Playlist Generator Project

## Overview
Traditional music recommendation systems excel at identifying patterns in a user's listening history but often fail to adapt to real-time mood or context. This limitation leads to repetitive and irrelevant recommendations that reduce user satisfaction [Mattew, 2024]. Our Personalized Music Playlist Generator is designed to address this challenge by combining Natural Language Processing, Machine Learning models, and Reinforcement Learning to create adaptive, context-aware playlists. Using data from the [Spotify API](https://developer.spotify.com/documentation/web-api), [Genius API](https://docs.genius.com/), and [Kaggle Dataset](https://www.kaggle.com/datasets/bwandowando/spotify-songs-with-attributes-and-lyrics), the system analyzes song lyrics and user interactions to understand emotional tone and thematic content.

This repository contains all of the code necessary to fetch the song data, preprocess it, run the ML model, train the RL model, and finally interact with the RL model in the UI. Running the ML portion requires performing the "Run ML Model" steps below. Fetching data, preprocessing, training, and running the RL model is more intensive, and a pre-trained model has been provided for this purpose. As such, you only need to perform the steps under "Run RL".

## Installation
This project utilizes Poetry and Docker to simplify the process of fetching data, running the training, and finally running the RL Model in the UI. Ensure that you have [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed to be able to run the various necessary Docker Compose commands. All docker compose commands should be executed in the root directory of the repository, and all file paths are with respect to the root directory.

### Run ML Model
Begin by downloading the [Kaggle Dataset](https://www.kaggle.com/datasets/bwandowando/spotify-songs-with-attributes-and-lyrics). Extract it, and place it at 

```bash
playlistgenerator/Datasets/spotify_songs.csv
```

Execute the preprocessing portion:

```bash
docker compose up run-ml-model-preprocess --build
```

Execute the sentiment analysis portion:

```bash
docker compose up run-ml-model-sentiment --build
```

Within `playlistgenerator/ModelBuildingKaggle/model.py`, you can tweak the constant `INDEX` to select the song used as the basis for the generated recommendations. Execute the following to generate the top recommendations based on this song:   

```bash
docker compose up run-ml-model-reccs --build
```

### Fetch Spotify Data
To fetch the data from the Spotify and Genius APIs, you must first create APIs tokens and credentials for both of these platforms. [Spotify API](https://developer.spotify.com/documentation/web-api), [Genius API](https://docs.genius.com/). Once created, these credentials should be added in `playlistgenerator/FetchSpotify/data.py`, under the variables of `client_id`, `client_secret`, `redirect_uri`, `genius_token`.

Execute the spotify fetching portion:

```bash
docker compose up run-spotify-fetch --build
```

### Train RL
After executing all of the prior steps, we have now generated recommendations from the Kaggle Dataset, and fetched a user's spotify song history. These will be combined to train the RL model. Hyperparameters for this process, such as the batch size and number of iterations, can be tweaked in `playlistgenerator/ReinforcementLearning/constants.py`

To execute training, run the following:
```bash
docker compose up run-pipeline --build
```

This process is quite lengthy, and thoroughly training the model with a large number of iterations and songs will take multiple days. The training process will incrementally provide updates on progress, and save the trained policy in case that training crashes or halts. 


### Run RL
Finally, to interact with the RL model in the UI, execute the following. 

```bash
docker compose up run-webapp --build
```

Note that due to issues with Spotify account flagging due to suspicious activity, the spotify account credentials provided in the UI is not used. Instead, the data fetched from the `Fetch Spotify Data` step for a particular user is leveraged when making recommendations. 
