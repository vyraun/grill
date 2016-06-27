import gym
from grill.qfunction import TabularQFunction
from grill.policy import QGreedyPolicy, EpsilonGreedyPolicy
from grill.method import QLearning
from grill.core import Engine


env = gym.make('FrozenLake-v0')
qfunction = TabularQFunction(env)
qgreedy = QGreedyPolicy(qfunction)
policy = EpsilonGreedyPolicy(qgreedy, 0.1)
qlearning = QLearning(qfunction)
trainer = Engine()
qlearning.register_with_engine(trainer)
trainer.run(policy, num_episodes=10000, render=False)
player = Engine()
player.run(qgreedy, num_episodes=100)
