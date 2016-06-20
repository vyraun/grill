from qfunction import DiscreteQFunction
from grill.util.misc import superhash
from collections import defaultdict
import numpy as np

class TabularQFunction(DiscreteQFunction):
    def __init__(self, env, default_value = 0.0):
        super(TabularQFunction, self).__init__(env)
        self._table = defaultdict(lambda: default_value)

    def get(self, observation, action):
        return self._table[(observation, action)]

    def set(self, observation, action, value):
        self._table[(observation, action)] = value
