from gradient_method import GradientMethod
from grill.util.memory import Memory
from grill.util.misc import sanity_check_params, pickle_copy, keywise
import numpy as np
import theano.tensor as T


class DeepQLearning(GradientMethod):
    def __init__(
            self, qfunc,
            update_fn,
            batchsize=32,
            N=1000000,  # Memory replay capacity
            C=10000     # How many iterations between resetting target network
    ):
        super(DeepQLearning, self).__init__('DQN', update_fn, batchsize)
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
        if engine.itr % self.C == 0:
            self._target = pickle_copy(self.qfunc)
            print 'Cloning target network'

        observation, action, next_observation, reward = episode.latest_transition()
        terminal = episode.done
        self._memory.add((observation, action, reward, next_observation, terminal))
        batch = self._memory.sample(self.batchsize)
        observations, actions, rewards, next_observations, terminals = keywise(batch, range(5))
        ys = rewards + (1-terminals) * engine.discount * self._target.best_values(next_observations)
        self._update(np.stack(observations), np.array(actions, dtype='int32'), np.array(ys))
        sanity_check_params(self.qfunc.get_params())
