import os
import numpy as np
import theano.tensor as T

DEFAULT_EPSILON = 1e-8

def superhash(obj):
    if isinstance(obj, np.ndarray):
        array = obj.copy()
        array.flags.writeable = False
        return hash(array.data)
    else:
        return hash(obj)

def safelog(x, epsilon=DEFAULT_EPSILON):
    return np.log(x + epsilon)

def safelog_sym(x, epsilon=DEFAULT_EPSILON):
    return T.log(x + epsilon)

def add_dim(array, axis=0):
    shape = list(array.shape)
    shape.insert(axis, 1)
    return array.reshape(shape)

def sanity_check_params(params):
    for param in params:
        assert not np.any(np.isnan(param))

# Intended to manufacture callbacks
def param_saver(parametric, filename='params.npz', frequency=1):
    def save(engine, _=None):
        itr = engine.itr
        if itr % frequency == 0:
            parametric.save_params(os.path.join(engine.log_dir, filename))
    return save

def keywise_cat(dicts, keys):
    return [np.concatenate([d[key] for d in dicts]) for key in keys]

def attrwise_cat(objects, keys):
    return [np.concatenate([getattr(o, key) for o in objects]) for key in keys]
