import gym
import grill
from grill.implementation import MultilayerPerceptron
from grill.policy import ParametricPolicy
from grill.baseline import ZeroBaseline
from grill.method import VanillaPolicyGradient
from grill.util.misc import param_logger, load_logged_params
from grill.util.preprocess import flatten
from grill.core import Engine
import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL


game = 'MountainCar'
log_dir = '/workspace/grill/log/vpg-' + game + '/'
env = gym.make(game + '-v0')

mlp = MultilayerPerceptron([2, 100, 3], output_nl=NL.softmax)
load_logged_params(mlp)

policy = ParametricPolicy(env, mlp, 'stochastic')
trainer = Engine(log_dir=log_dir)
vpg = VanillaPolicyGradient(policy, ZeroBaseline())
vpg.register_with_engine(trainer)
trainer.register_callback('post-episode', 'save', param_logger(mlp))
trainer.run(policy, phi=flatten, num_episodes=1000, render=True)

player = Engine()
player.run(policy, phi=flatten, num_episodes=25, render=True)
