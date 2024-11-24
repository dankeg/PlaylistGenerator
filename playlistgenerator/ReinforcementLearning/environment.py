from __future__ import absolute_import, division, print_function

import random

import numpy as np
from playlistgenerator.ReinforcementLearning.utils import softmax
import pandas as pd

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class MusicPlaylistEnv(py_environment.PyEnvironment):
    def __init__(self, data):
        # Define action and observation specifications
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=4, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(30,), dtype=np.float64, minimum=0, name='observation')
        
        # Save the initial data
        self.saved_data = data

        # Copy the data and add unique_id and user_score columns
        self.data = self.saved_data.copy()
        self.data['unique_id'] = range(1, len(self.saved_data) + 1)
        self.data['user_score'] = 0

        # Hash track_name and artist_name for unique identification
        self.data["track_name"] = self.data["track_name"].apply(lambda x: hash(x))
        self.data["artist_name"] = self.data["artist_name"].apply(lambda x: hash(x))

        # Initialize the state with the first 5 rows of data
        self._state = self.data.head(5)
        self._episode_ended = False
        self._current_time_step = self._reset()

        self.count = 0
        
    def action_spec(self):
        # Return the action specification
        return self._action_spec

    def observation_spec(self):
        # Return the observation specification
        return self._observation_spec

    def _reset(self):
        # Reset the environment to the initial state
        self.count = 0
        self._episode_ended = False

        # Reset the data and add unique_id and user_score columns
        self.data = self.saved_data.copy()
        self.data['unique_id'] = range(1, len(self.saved_data) + 1)
        self.data['user_score'] = 0

        # Hash track_name and artist_name for unique identification
        self.data["track_name"] = self.data["track_name"].apply(lambda x: hash(x))
        self.data["artist_name"] = self.data["artist_name"].apply(lambda x: hash(x))

        # Initialize the state with the first 5 rows of data
        self._state = self.data.head(5)

        # Return the initial time step
        return ts.restart(self._state.to_numpy().flatten())

    def _step(self, action):
        # If the episode has ended, reset the environment
        if self._episode_ended:
            return self.reset()
        
        # Increment the step count
        print(f"Count {self.count}")

        # If the step count exceeds 300, terminate the episode
        if self.count > 30:
            self._episode_ended = True
            return ts.termination(self._state.to_numpy().flatten(), 0)

        # Get the user scores and compute the softmax output
        score_array = self._state["user_score"].to_numpy()
        softmax_output = softmax(score_array)

        # Determine if the action is selected based on the softmax probability
        probability = softmax_output[action]
        selection = random.random() < probability

        if selection:
            # If the action is selected, update the user score and reorder the data
            id_array = self._state["unique_id"].to_numpy()
            id = id_array[action]
            self._state.loc[self._state['unique_id'] == id, 'user_score'] += 1
            self.data.loc[self.data['unique_id'] == id, 'user_score'] += 1

            # Swap rows and move the selected row to the end
            self.data.iloc[[4, 5]] = self.data.iloc[[5, 4]].copy().values
            row_to_move = self.data.iloc[5:6].copy()
            self.data = pd.concat([self.data.drop(self.data.index[5]), row_to_move], ignore_index=True)

            # Update the state with the first 5 rows of data
            self._state = self.data.iloc[:5].copy()

            # Return a transition time step with a reward of 1.0
            return ts.transition(self._state.to_numpy().flatten(), reward=1.0, discount=.99)
        else:
            # If the action is not selected, update the user score and reorder the data
            id_array = self._state["unique_id"].to_numpy()
            id = id_array[action]
            self._state.loc[self._state['unique_id'] == id, 'user_score'] += -1
            self.data.loc[self.data['unique_id'] == id, 'user_score'] += -1

            # Swap rows and move the selected row to the end
            self.data.iloc[[action, 5]] = self.data.iloc[[5, action]].copy().values
            row_to_move = self.data.iloc[5:6].copy()
            self.data = pd.concat([self.data.drop(self.data.index[5]), row_to_move], ignore_index=True)

            # Update the state with the first 5 rows of data
            self._state = self.data.iloc[:5].copy()

            # Return a transition time step with a reward of 0.0
            return ts.transition(self._state.to_numpy().flatten(), reward=0.0, discount=.99)
