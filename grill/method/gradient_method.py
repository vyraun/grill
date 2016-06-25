from method import Method
import theano
import theano.tensor as T


# Base class for gradient-based optimization methods
class GradientMethod(Method):
    def __init__(self, name, update_fn, batchsize):
        super(GradientMethod, self).__init__(name)
        self.update_fn = update_fn
        self.batchsize = batchsize

    def _build_update(self, inputs, param_vars, loss_var, outputs=[]):
        # grads = T.grad(loss_var, param_vars)
        # self._update = theano.function(
        #         inputs=inputs, outputs=outputs,
        #         updates=tuple((param, param - self.learning_rate*grads[i])
        #                 for i, param in enumerate(param_vars)),
        #         allow_input_downcast=True
        # )
        update_info = self.update_fn(loss_var, param_vars)
        self._update = theano.function(
                inputs=inputs, outputs=outputs,
                updates=update_info,
                allow_input_downcast=True
        )
