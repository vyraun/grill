from grill.theano import TheanoFunction
import theano.tensor as T
import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL

class ConvolutionalNetwork(TheanoFunction):
    def __init__(self, input_shape, num_out,
            filters=[[32, 8], [64, 4], [64, 3]],
            poolings=[4, 2, 1],
            conv_nl=NL.rectify,
            hidden_sizes=[512],
            fc_nl=NL.rectify,
            output_nl=None  # Change to softmax to get probabilities
    ):
        assert len(input_shape) == 3

        self._input_var = T.ftensor4('input')

        l_prev = L.InputLayer(
            shape=(None,) + input_shape,
            input_var=self._input_var
        )

        for filter, pooling in zip(filters, poolings):
            l_prev = L.Conv2DLayer(l_prev, filter[0], filter[1],
                    nonlinearity=conv_nl)
            l_prev = L.MaxPool2DLayer(l_prev, pooling)

        for size in hidden_sizes:
            l_prev = L.DenseLayer(l_prev,
                num_units=size,
                nonlinearity=fc_nl
            )

        l_output = L.DenseLayer(l_prev,
            num_units=num_out,
            nonlinearity=output_nl,
            name="output"
        )

        self._output_var = L.get_output(l_output)
        self._param_vars = L.get_all_params(l_output, trainable=True)

        super(ConvolutionalNetwork, self).__init__()
