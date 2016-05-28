from grill.theano import TheanoFunction
import theano.tensor as T
import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL

class MultilayerPerceptron(TheanoFunction):
    def __init__(self, sizes,
            nl=NL.rectify,
            output_nl=None  # Change to softmax to get probabilities
    ):
        self._input_var = T.fmatrix('input')

        l_prev = L.InputLayer(
            shape=(None, sizes[0]),
            input_var=self._input_var
        )

        for size in sizes[1:-1]:
            l_prev = L.DenseLayer(l_prev,
                num_units=size,
                nonlinearity=nl
            )

        l_output = L.DenseLayer(l_prev,
            num_units=sizes[-1],
            nonlinearity=output_nl,
            name="output"
        )

        self._output_var = L.get_output(l_output)
        self._param_vars = L.get_all_params(l_output, trainable=True)

        super(MultilayerPerceptron, self).__init__()
