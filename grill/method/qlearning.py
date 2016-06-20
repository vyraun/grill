from method import Method

class QLearning(Method):
    def __init__(self, qfunc, learning_rate=0.1):
        super(QLearning, self).__init__('Q-learning')
        self.qfunc = qfunc
        self.learning_rate = learning_rate

    def post_step(self, engine, episode):
        observation, action, next_observation, reward = episode.latest_transition()
        terminal = episode.done
        current_q = self.qfunc.get(observation, action)
        value = current_q + self.learning_rate * \
                (reward + engine.discount * self.qfunc.best_value(next_observation) - current_q)
        self.qfunc.set(observation, action, value)
