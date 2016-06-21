import config as cfg
import numpy as np

def discounted_partial_sums(rewards, discount):
    return np.cumsum(np.array(rewards) * discount**np.arange(len(rewards)))

def discounted_sum(rewards, discount):
    return np.sum(np.array(rewards) * discount**np.arange(len(rewards)))

# Generalized advantage estimation
# Adapted from John Schulman's implementation
# (https://github.com/joschu/modular_rl)
def estimate_advantages(baseline, episodes, gamma, lam):
    # Compute return, baseline, advantage
    for episode in episodes:
        b = baseline.predict(episode)
        b1 = np.append(b, 0 if episode.done else b[-1])
        deltas = episode.rewards + gamma*b1[1:] - b1[:-1]
        episode.advantages = discounted_partial_sums(deltas, gamma * lam)
    alladv = np.concatenate([episode.advantages for episode in episodes])
    # Standardize advantage
    std = alladv.std()
    mean = alladv.mean()
    for episode in episodes:
        episode.advantages = (episode.advantages - mean) / (std + cfg.DEFAULT_EPSILON)

# def estimate_advantages(vf, paths, gamma, lam):
#     # Compute return, baseline, advantage
#     for path in paths:
#         path["return"] = discount(path["reward"], gamma)
#         b = path["baseline"] = vf.predict(path)
#         b1 = np.append(b, 0 if path["terminated"] else b[-1])
#         deltas = path["reward"] + gamma*b1[1:] - b1[:-1]
#         path["advantage"] = discount(deltas, gamma * lam)
#     alladv = np.concatenate([path["advantage"] for path in paths])
#     # Standardize advantage
#     std = alladv.std()
#     mean = alladv.mean()
#     for path in paths:
#         path["advantage"] = (path["advantage"] - mean) / std
