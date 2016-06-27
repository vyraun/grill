from qfunction import DiscreteQFunction
from grill.theano import TheanoContainer
from grill.util.misc import add_dim
import numpy as np

# Skeleton for function approximation. We assume that the implementation's call
# operator produces all Q-values for several observations at a time
class ParametricQFunction(DiscreteQFunction, TheanoContainer):
    def __init__(self, env, implementation):
        super(ParametricQFunction, self).__init__(env)
        TheanoContainer.__init__(self, implementation)

    def get_all_multiple(self, observations):
        return self.implementation(observations)

    def get_all(self, observation):
        return self.get_all_multiple(add_dim(observation))[0]

    def best_values(self, observations):
        return np.max(self.get_all_multiple(observations), axis=1)

    def best_actions(self, observations):
        return np.argmax(self.get_all_multiple(observations), axis=1)
