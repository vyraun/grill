import gym
import grill
from grill.implementation import ConvolutionalNetwork
from grill.policy import ParametricPolicy
from grill.baseline import ZeroBaseline
from grill.method import VanillaPolicyGradient
from grill.util.misc import param_saver
from grill.util.preprocess import atari_preprocessor
from grill.core import Engine
import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL


game = 'Breakout'
log_dir = '/Users/garrett/BoxSync/github/grill/log/vpg-' + game + '/'
m = 4
size = (84,84)
env = gym.make(game + '-v0')
preprocess = atari_preprocessor(m=m, size=size)

convnet = ConvolutionalNetwork((None, m, size[0], size[1]), env._n_actions,
        output_nl=NL.softmax)
try:
    convnet.load_params(log_dir + 'params.npz')
    print 'Successfully loaded network weights from file'
except:
    print 'Failed to load network weights from file'

policy = ParametricPolicy(env, convnet, 'stochastic')
trainer = Engine(log_dir=log_dir)
vpg = VanillaPolicyGradient(policy, ZeroBaseline())
vpg.register_with_engine(trainer)
trainer.register_callback('post-episode', 'save', param_saver(convnet))
trainer.run(policy, phi=preprocess, num_episodes=10000, render=True)

player = Engine()
player.run(policy, phi=preprocess, num_episodes=25, render=True)
