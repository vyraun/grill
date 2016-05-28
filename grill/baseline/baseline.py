import numpy as np


class Baseline(object):
    def fit(self, engine, episode):
        raise NotImplementedError

    def predict(self, episode):
        raise NotImplementedError


class ZeroBaseline(Baseline):
    def fit(self, engine, episode):
        pass

    def predict(self, episode):
        return np.zeros_like(episode.rewards)
