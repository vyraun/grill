from method import Method
from grill.util.memory import ReplayMemory
import numpy as np
import theano
import theano.tensor as T

# Assumes that policy is an ApproximateQFunction
class DeepQLearning(Method):
    def __init__(
            self, policy, qfunc,
            discount=0.99,
            learning_rate=0.001,
            batchsize=32,
            N=1000000, # Memory replay capacity
            C=10000, # How many iterations between resetting target network
            m=4, # How many recent frames to pass to network
            k=4 # How many times to repeat the current action (for frame skipping)
    ):
        super(DeepQLearning, self).__init__(policy, discount)
        self.qfunc = qfunc
        self.learning_rate = learning_rate
        self.batchsize = batchsize
        self.N = N 
        self.C = C 
        self.m = m 
        self.k = k
        self.memory = ReplayMemory(N)

        obs_var = self.qfunc.get_input_var()
        action_var = T.iscalar('action')
        y_var = T.dscalar('y')

        params = self.qfunc.get_param_vars()
        q_vals_var = self.qfunc.get_output_var()
        loss_var = (y_var - q_vals_var[action_var])**2
        grads = T.grad(loss_var, params)
        self._sgd = theano.function(
                inputs=[obs_var, action_var, y_var],
                updates=tuple((param, param - self.learning_rate*grads[i])
                        for i, param in enumerate(params)))

    def get_action(self, observation, t):
        if not hasattr(self, '_recent_obs'):
            self._recent_obs = [observation.copy() for _ in range(self.m)]

        if t % self.k == 0 or not hasattr(self, '_current_action'):
            self._current_action = self.policy.get_action(self._prep_input())

        return self._current_action

    def post_step(self, observation, action, next_observation, reward, done, t, total_t, **kwargs):
        prev_input = self._prep_input()
        if done:
            y = reward
        else:
            self._update_recent_obs(next_observation)
            current_params = self.qfunc.get_params()
            if total_t % self.C == 0:
                self._target_params = current_params
            self.qfunc.set_params(self._target_params)
            best_q = self.qfunc.best_value(self._prep_input())
            self.qfunc.set_params(current_params)
            y = reward + self.discount * best_q
        
        self._sgd(prev_input, action, y)

    def _update_recent_obs(self, new_obs):
        new_obs = np.maximum(self._recent_obs[0], new_obs)
        self._recent_obs.pop()
        self._recent_obs.insert(0, new_obs)

    def _prep_input(self):
        stacked_input = np.stack(self._recent_obs)
        return stacked_input.reshape((1,) + stacked_input.shape)