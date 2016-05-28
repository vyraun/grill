import gym
import grill
from grill.implementation import ConvolutionalNetwork
from grill.qfunction import ParametricQFunction
from grill.policy import QGreedyPolicy, EpsilonGreedyPolicy
from grill.method import DeepQLearning
from grill.util.misc import sanity_check_params, param_saver
from grill.util.preprocess import atari_preprocessor
from grill.core import Engine
import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL


def update_epsilon(engine, _):
    itr = engine.state.itr
    policy.epsilon = max(1.0-itr/100000.0, 0.1)

game = 'Breakout'
log_dir = '/Users/garrett/Box Sync/github/grill/log/dqn-' + game + '/'
m = 4
size = (84,84)
env = gym.make(game + '-v0')
preprocess = atari_preprocessor(m=m, size=size)

convnet = ConvolutionalNetwork((None, m, size[0], size[1]), env._n_actions)
try:
    convnet.load_params(log_dir + 'params.npz')
    print 'Successfully loaded network weights from file'
except:
    print 'Failed to load network weights from file'

qfunction = ParametricQFunction(env, convnet)
qgreedy = QGreedyPolicy(qfunction)
policy = EpsilonGreedyPolicy(qgreedy, epsilon=0.5)
dqn = DeepQLearning(qfunction)
trainer = Engine(log_dir=log_dir)
dqn.register_with_engine(trainer)
trainer.register_callback('post-step', 'save', param_saver(convnet))
#trainer.register_callback('pre-step', 'update epsilon', update_epsilon)
trainer.run(policy, phi=preprocess, num_episodes=1000, render=True)

player = Engine()
player.run(EpsilonGreedyPolicy(qgreedy, epsilon=0.05), phi=preprocess, num_episodes=25, render=True)
