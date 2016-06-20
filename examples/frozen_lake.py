import gym
import grill
from grill.baseline import ZeroBaseline
from grill.implementation import MultilayerPerceptron
from grill.qfunction import TabularQFunction, ParametricQFunction
from grill.policy import QGreedyPolicy, EpsilonGreedyPolicy, ParametricPolicy
from grill.method import QLearning, VanillaPolicyGradient
from grill.core import Engine
from grill.util.preprocess import onehot_encoder, identity
import lasagne.nonlinearities as NL


env = gym.make('FrozenLake-v0')
# qfunction = TabularQFunction(env)
# qgreedy = QGreedyPolicy(qfunction)
# policy = EpsilonGreedyPolicy(qgreedy, 0.1)
# qlearning = QLearning(qfunction)
phi = identity
mlp = MultilayerPerceptron([env.observation_space.n, env.action_space.n],
        output_nl=NL.softmax)
policy = ParametricPolicy(env, mlp, 'stochastic')
vpg = VanillaPolicyGradient(policy, ZeroBaseline())
phi = onehot_encoder(env.observation_space.n)
trainer = Engine()
# qlearning.register_with_engine(trainer)
vpg.register_with_engine(trainer)
# trainer.register_callback('pre-step', 'update epsilon', update_epsilon)
trainer.run(policy, phi=phi, num_episodes = 10000, render=False)
player = Engine()
player.run(qgreedy, num_episodes = 100)
