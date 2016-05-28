import numpy as np
from scipy.misc import imresize

def identity(raw_observation, _=None):
    return raw_observation

def flatten(raw_observation, _=None):
    return raw_observation.flatten()

def luminance(raw_observation, _=None):
    r, g, b = raw_observation[:,:,0], raw_observation[:,:,1], raw_observation[:,:,2]
    return 0.2126*r + 0.7152*g + 0.0722*b

def resizer(*size):
    return lambda raw_observation, _=None: imresize(raw_observation, size)

# Implements preprocessing method described in the paper by Mnih, et al.
def atari_preprocessor(m=4, size=(84,84)):
    def preprocess(raw_observation, engine):
        raw_observations = engine.latest_episode().raw_observations

        # Make sure there are enough frames (duplicate latest if not)
        latest = raw_observation
        while len(raw_observations) < m+1:
            # This assignment is safe; a new list is created
            raw_observations = [latest] + raw_observations

        # Calculate luminance and resize
        recent_frames = []
        for i in range(m):
            maxed = np.maximum(raw_observations[-(1+i)], raw_observations[-(i+2)])
            luma = luminance(maxed)
            resized = imresize(luma, size)
            recent_frames.append(resized)

        # Stack and normalize pixel values (to avoid saturating neural net)
        return np.stack(recent_frames) / 255.0
    return preprocess
