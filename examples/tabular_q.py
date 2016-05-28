import gym
import grill
from grill.qfunctions import TabularQFunction
from grill.policies import QGreedyPolicy, EpsilonGreedyPolicy
from grill.methods import QLearning


env = gym.make('MsPacman-v0')
qfunction = TabularQFunction(env)
qgreedy = QGreedyPolicy(env, qfunction)
policy = EpsilonGreedyPolicy(env, qgreedy, 0.1)
qlearning = QLearning(qfunction)
grill.run(env, policy, qlearning, render=True)