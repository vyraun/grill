import sys
import gym
import grill
from grill.implementation import ConvolutionalNetwork
from grill.qfunction import ParametricQFunction
from grill.policy import QGreedyPolicy, EpsilonGreedyPolicy
from grill.method import DeepQLearning
from grill.util.misc import sanity_check_params, param_logger, load_params, show_matrix
from grill.util.preprocess import atari_preprocessor
from grill.core import Engine
import grill.core.config as cfg
import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL
import matplotlib.pyplot as plt


game = sys.argv[1]
cfg.LOG_DIR = '~/code/grill/log/dqn-' + game
eps_max = 1.0
eps_min = 0.1
eps_steps = 1e6

def update_epsilon(engine, _):
    itr = engine.itr
    # policy.epsilon = max(1.0-itr/1000000.0, 0.1)
    annealed = eps_max - (eps_max - eps_min) * (itr / eps_steps)
    policy.epsilon = max(annealed, eps_min)
    # print policy.epsilon


m = 4
size = (84,84)
filters = [[32, 8], [64, 4], [64, 3]]
poolings = [4, 2, 1]
env = gym.make(game + '-v0')
preprocess = atari_preprocessor(m=m, size=size)

convnet = ConvolutionalNetwork(
        input_shape=(m, size[0], size[1]),
        num_out=env.action_space.n,
        filters=filters,
        poolings=poolings)
load_params(convnet)

qfunction = ParametricQFunction(env, convnet)
qgreedy = QGreedyPolicy(qfunction)
policy = EpsilonGreedyPolicy(qgreedy, epsilon=0.5)
dqn = DeepQLearning(qfunction,
        N=10000,
        update_fn=lasagne.updates.adam)
        # update_fn=lambda loss, params: lasagne.updates.rmsprop(loss, params,
        #         learning_rate=0.00025,
        #         rho=0.95,
        #         epsilon=0.01))
trainer = Engine(episode_memory_size=10)
dqn.register_with_engine(trainer)
trainer.register_callback('post-step', 'save', param_logger(convnet))
trainer.register_callback('pre-step', 'update epsilon', update_epsilon)
# trainer.register_callback('post-step', 'show observation', show_obs)
player = Engine(episode_memory_size=10)

returns = []
for epoch in range(100):
    trainer.run(policy,
            phi=preprocess,
            num_episodes=10,
            render=False)

    avg_return = np.mean(player.run(EpsilonGreedyPolicy(qgreedy, epsilon=0.05),
            phi=preprocess,
            num_episodes=10,
            render=False))
    print 'Epoch', epoch, ':', avg_return
    returns.append(avg_return)

plt.figure()
plt.plot(returns)
plt.show()
