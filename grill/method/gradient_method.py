from method import Method
import theano
import theano.tensor as T


# Base class for gradient-based optimization methods
class GradientMethod(Method):
    def __init__(self, name, learning_rate, batchsize):
        super(GradientMethod, self).__init__(name)
        self.learning_rate = learning_rate
        self.batchsize = batchsize

    def _build_update(self, inputs, param_vars, loss_var, outputs=[]):
        grads = T.grad(loss_var, param_vars)
        self._update = theano.function(
                inputs=inputs, outputs=outputs,
                updates=tuple((param, param - self.learning_rate*grads[i])
                        for i, param in enumerate(param_vars)),
                allow_input_downcast=True
        )
