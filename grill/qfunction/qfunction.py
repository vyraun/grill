import numpy as np
from gym.spaces import Discrete


class QFunction(object):
    def __init__(self, env):
        self.env = env

    def get(self, observation, action):
        raise NotImplementedError

    def set(self, observation, action, value):
        raise NotImplementedError

    # max over actions
    def best_value(self, observation):
        raise NotImplementedError

    # argmax over actions
    def best_action(self, observation):
        raise NotImplementedError


# Operates under the assuption that the env has Discrete action space
class DiscreteQFunction(QFunction):
    def __init__(self, env):
        assert isinstance(env.action_space, Discrete)
        super(DiscreteQFunction, self).__init__(env)
        self._num_actions = env.action_space.n

    # This can be overridden for functions that produce many Q-values at once
    # (e.g. neural networks)
    def get_all(self, observation):
        return [self.get(observation, action) for action in range(self._num_actions)]

    def best_value(self, observation):
        return np.max(self.get_all(observation))

    def best_action(self, observation):
        return np.argmax(self.get_all(observation))
