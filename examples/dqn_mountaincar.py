import gym
import grill
from grill.implementation import MultilayerPerceptron
from grill.qfunction import ParametricQFunction
from grill.policy import QGreedyPolicy, EpsilonGreedyPolicy
from grill.method import DeepQLearning
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
cfg.LOG_DIR = '~/code/grill/log/dqn-' + game
env = gym.make(game + '-v0')

mlp = MultilayerPerceptron([2, 100, 3])
load_logged_params(mlp)

qfunction = ParametricQFunction(env, mlp)
qgreedy = QGreedyPolicy(qfunction)
policy = EpsilonGreedyPolicy(qgreedy, epsilon=0.1)
dqn = DeepQLearning(policy, qfunction)
trainer = Engine(log_dir=log_dir)

def update_epsilon(engine, _):
    itr = engine.state.itr
    policy.epsilon = max(1.0-itr/1000000.0, 0.1)

dqn.register_with_engine(trainer)
trainer.register_callback('post-step', 'save', param_logger(mlp))
trainer.register_callback('pre-step', 'update epsilon', update_epsilon)
#trainer.run(policy, phi=flatten, num_episodes=1000, render=True)

player = Engine()
player.run(qgreedy, phi=flatten, num_episodes=25, render=True)
