from theano_function import TheanoFunction

# Base class for parametric objects (e.g. policies, Q-functions) that accept a
# Theano function as implementation and forward all calls to it
class TheanoContainer(TheanoFunction):
    def __init__(self, implementation):
        self.implementation = implementation

    def get_input_var(self):
        return self.implementation.get_input_var()

    def get_output_var(self):
        return self.implementation.get_output_var()

    def get_param_vars(self):
        return self.implementation.get_param_vars()

    def get_params(self):
        return self.implementation.get_params()

    def set_params(self, params):
        self.implementation.set_params(params)

    def __call__(self, *args):
        return self.implementation(*args)
