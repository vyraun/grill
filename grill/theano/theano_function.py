from grill.util.misc import sanity_check_params
import numpy as np
import theano

# Base class for functions implemented in Theano, which are used to parameterize
# Q-functions, policies, etc. See the implementation module for subclasses
# (e.g. neural networks)
class TheanoFunction(object):
    # Override this. Init should set up member variables _input_var, _output_var,
    # and _param_vars, then call super init
    def __init__(self):
        self._fn = theano.function(
            inputs=[self._input_var],
            outputs=[self._output_var],
            allow_input_downcast=True
        )

    def get_input_var(self):
        return self._input_var

    def get_output_var(self):
        return self._output_var

    def get_param_vars(self):
        return self._param_vars

    def get_params(self):
        return [param_var.eval() for param_var in self.get_param_vars()]

    def set_params(self, params):
        assert len(params) == len(self._param_vars)
        sanity_check_params(params)
        for param_var, new_value in zip(self.get_param_vars(), params):
            param_var.set_value(new_value)

    def save_params(self, filename):
        with open(filename, 'wb') as f:
            np.savez(f, *self.get_params())

    def load_params(self, filename):
        with np.load(filename) as data:
            self.set_params([data['arr_'+str(i)] for i in range(len(data.files))])

    def __call__(self, *args):
        return self._fn(*args)[0]
