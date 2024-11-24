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
        self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=4, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(30,), dtype=np.float64, minimum=0, name='observation')
        
        self.saved_data = data

        self.data = self.saved_data.copy()
        self.data['unique_id'] = range(1, len(self.saved_data) + 1)
        
        self.data['user_score'] = 0

        self.data["track_name"] = self.data["track_name"].apply(lambda x: hash(x))
        self.data["artist_name"] = self.data["artist_name"].apply(lambda x: hash(x))

        self._state = self.data.head(5)
        self._episode_ended = False
        self._current_time_step = self._reset()

        self.count = 0
        
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.count = 0
        self._episode_ended = False

        self.data = self.saved_data.copy()
        self.data['unique_id'] = range(1, len(self.saved_data) + 1)
        
        self.data['user_score'] = 0

        self.data["track_name"] = self.data["track_name"].apply(lambda x: hash(x))
        self.data["artist_name"] = self.data["artist_name"].apply(lambda x: hash(x))

        self._state = self.data.head(5)

        print(self._state.to_numpy().flatten())
        print(f"Len is {self._state.to_numpy().flatten().size}")
        return ts.restart(self._state.to_numpy().flatten())

    def _step(self, action):
        if self._episode_ended:
            return self.reset()
        
        self.count += 1
        if self.count > 300:
            return ts.termination(self._state.to_numpy().flatten(), 0)

        score_array = self._state["user_score"].to_numpy()
        softmax_output = softmax(score_array)
        print(f"The action is {action}")
        print(f"The softmax output is {softmax_output}")
        probability = softmax_output[action]
        selection = random.random() < probability

        if selection:
            id_array = self._state["unique_id"].to_numpy()
            id = id_array[action]
            self._state.loc[self._state['unique_id'] == id, 'user_score'] += 1
            self.data.loc[self.data['unique_id'] == id, 'user_score'] += 1

            self.data.iloc[[4, 5]] = self.data.iloc[[5, 4]].copy().values
            row_to_move = self.data.iloc[5:6].copy()
            self.data = pd.concat([self.data.drop(self.data.index[5]), row_to_move], ignore_index=True)

            self._state = self.data.iloc[:5].copy()
            print(self._state.to_numpy().flatten())
            print(f"Len is {self._state.to_numpy().flatten().size}")
            return ts.transition(self._state.to_numpy().flatten(), reward=1.0, discount=.99)
        else:
            id_array = self._state["unique_id"].to_numpy()
            id = id_array[action]
            self._state.loc[self._state['unique_id'] == id, 'user_score'] += -1
            self.data.loc[self.data['unique_id'] == id, 'user_score'] += -1

            self.data.iloc[[action, 5]] = self.data.iloc[[5, action]].copy().values
            row_to_move = self.data.iloc[5:6].copy()
            self.data = pd.concat([self.data.drop(self.data.index[5]), row_to_move], ignore_index=True)

            self._state = self.data.iloc[:5].copy()
            print(f"Len is {self._state.to_numpy().flatten().size}")
            return ts.transition(self._state.to_numpy().flatten(), reward=0.0, discount=.99)

