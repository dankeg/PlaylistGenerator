from __future__ import absolute_import, division, print_function

import random
from copy import deepcopy

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class MusicPlaylistEnv(py_environment.PyEnvironment):
    def __init__(self):
        pass

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        pass
        return ts.restart(np.array(output, dtype=np.int32))

    def _step(self, action):
        pass
        return ts.transition(np.array(output, dtype=np.int32), reward, discount=1.0)