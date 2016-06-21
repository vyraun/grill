from gradient_method import GradientMethod
from grill.util.memory import Memory
from grill.util.misc import add_dim, sanity_check_params
import numpy as np
import theano.tensor as T


class DeepQLearning(GradientMethod):
    def __init__(
            self, qfunc,
            learning_rate=0.001,
            batchsize=32,
            N=1000000,  # Memory replay capacity
            C=10000     # How many iterations between resetting target network
    ):
        super(DeepQLearning, self).__init__('DQN', learning_rate, batchsize)
        self.qfunc = qfunc
        self.N = N
        self.C = C
        self._memory = Memory(N)

        observation_var = self.qfunc.get_input_var()
        action_var = T.ivector('action')
        y_var = T.fvector('y')

        qs_var = self.qfunc.get_output_var()
        relevant_qs_var = qs_var[T.arange(self.batchsize),action_var]
        loss_var = T.sum((y_var - relevant_qs_var)**2)/self.batchsize
        self._build_update(
                [observation_var, action_var, y_var],
                qfunc.get_param_vars(),
                loss_var
        )

    def post_step(self, engine, episode):
        observation, action, next_observation, reward = episode.latest_transition()
        terminal = episode.done
        self._memory.add((observation, action, reward, next_observation, terminal))
        batch = self._memory.sample(self.batchsize)
        observations, actions, ys = [], [], []
        for obs, a, r, next_obs, done in batch:
            if done:
                y = reward
            else:
                current_params = self.qfunc.get_params()
                if engine.itr % self.C == 0:
                    self._target_params = current_params
                self.qfunc.set_params(self._target_params)
                best_q = self.qfunc.best_value(next_observation)
                self.qfunc.set_params(current_params)
                y = reward + engine.discount * best_q

            observations.append(obs)
            actions.append(a)
            ys.append(y)

        self._update(np.stack(observations), np.array(actions, dtype='int32'), np.array(ys))
        sanity_check_params(self.qfunc.get_params())
