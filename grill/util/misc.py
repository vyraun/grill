import cPickle as pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as T
import grill.core.config as cfg
from grill.util.file import logpath

def superhash(obj):
    if isinstance(obj, np.ndarray):
        array = obj.copy()
        array.flags.writeable = False
        return hash(array.data)
    else:
        return hash(obj)

def safelog(x, epsilon=cfg.DEFAULT_EPSILON):
    return np.log(x + epsilon)

def safelog_sym(x, epsilon=cfg.DEFAULT_EPSILON):
    return T.log(x + epsilon)

def add_dim(array, axis=0):
    shape = list(array.shape)
    shape.insert(axis, 1)
    return array.reshape(shape)

def sanity_check_params(params):
    for param in params:
        assert not np.any(np.isnan(param))

# Intended to manufacture callbacks
def param_logger(parametric, filename='params.npz', frequency=1):
    def save(engine, _=None):
        if engine.itr % frequency == 0:
            parametric.save_params(logpath(filename))
    return save

def load_params(parametric, filename='params.npz', onfail=None):
    try:
        parametric.load_params(logpath(filename))
        print 'Successfully loaded network weights from file'
    except:
        print 'Failed to load network weights from file'
        if onfail: onfail()

def keywise(dicts, keys):
    return [np.array([d[key] for d in dicts]) for key in keys]

def attrwise(objects, keys):
    return [np.array([getattr(o, key) for o in objects]) for key in keys]

def keywise_cat(dicts, keys):
    return [np.concatenate([d[key] for d in dicts]) for key in keys]

def attrwise_cat(objects, keys):
    return [np.concatenate([getattr(o, key) for o in objects]) for key in keys]

def show_matrix(m):
    plt.imshow(m)
    plt.show()

def pickle_copy(o):
    return pickle.loads(pickle.dumps(o))
