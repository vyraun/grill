Grill
=====

A reinforcement learning library, compatible with OpenAI Gym
------------------------------------------------------------

Reinforcement learning is a rapidly growing area of research, and Grill aims to be accordingly flexible. Grill's primary design goals are
 * To be compatible with [OpenAI Gym](https://gym.openai.com/)
 * To be easily extensible so that it can be used to test-drive new algorithms for research
 * To come with a variety of ready-made algorithms

Grill is written in Python.

**Please note that Grill is still under heavy development and is _not_ yet production-ready.**

Dependencies
------------
Grill stands on the shoulders of giants:
 * NumPy
 * Theano
 * Lasagne
 * OpenAI Gym (of course)

Just show me some code, already
-------------------------------
    import gym
    from grill.qfunction import TabularQFunction
    from grill.policy import QGreedyPolicy, EpsilonGreedyPolicy
    from grill.method import QLearning
    from grill.core import Engine
    
    env = gym.make('FrozenLake-v0')                             # Create the environment
    qfunction = TabularQFunction(env)                           # Instantiate a tabular Q function
    qgreedy = QGreedyPolicy(qfunction)                          # Create a policy that selects the greedy Q action every time
    policy = EpsilonGreedyPolicy(qgreedy, 0.1)                  # Wrap it in a policy that selects a random action w.p. epsilon
    qlearning = QLearning(qfunction)                            # Use Q-learning to train the agent
    trainer = Engine()                                          # Create an engine to perform the training
    qlearning.register_with_engine(trainer)                     # Install callbacks in the engine to perform the Q-learning
    trainer.run(policy, num_episodes=10000, render=False)       # Run and train the agent
    player = Engine()                                           # Create another engine just to play the learned policy
    player.run(qgreedy, num_episodes=100, render=True)          # Run the greedy policy
