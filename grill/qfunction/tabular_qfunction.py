from qfunction import DiscreteQFunction
from grill.util.misc import superhash
from collections import defaultdict
import numpy as np

class TabularQFunction(DiscreteQFunction):
    def __init__(self, env, default_value = 0.0):
        super(TabularQFunction, self).__init__(env)
        self._table = defaultdict(lambda: default_value)

    def get(self, observation, action, key=None):
        obs_key = key or superhash(observation)
        return self._table[(obs_key, action)]

    def set(self, observation, action, value, key=None):
        obs_key = key or superhash(observation)
        self._table[(obs_key, action)] = value

    # Technically didn't need to override this but it's much better not to
    # rehash for every action
    def get_all(self, observation):
        obs_key = superhash(observation)
        return [self.get(observation, action, obs_key) for action in range(self._num_actions)]