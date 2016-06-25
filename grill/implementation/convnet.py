from grill.theano import TheanoFunction
import theano.tensor as T
import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL

class ConvolutionalNetwork(TheanoFunction):
    def __init__(self, input_shape, num_out, filters, poolings,
            conv_nl=NL.rectify,
            hidden_sizes=[100],
            hidden_nl=NL.rectify,
            output_nl=None  # Change to softmax to get probabilities
    ):
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.num_out = num_out
        self.filters = filters
        self.poolings = poolings
        self.hidden_sizes = hidden_sizes
        self.hidden_nl = hidden_nl
        self.output_nl = output_nl

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
                nonlinearity=hidden_nl
            )

        l_output = L.DenseLayer(l_prev,
            num_units=num_out,
            nonlinearity=output_nl,
            name="output"
        )

        self._output_var = L.get_output(l_output)
        self._param_vars = L.get_all_params(l_output, trainable=True)

        super(ConvolutionalNetwork, self).__init__()


    # def clone(self):
    #     convnet = ConvolutionalNetwork(
    #             input_shape=self.input_shape,
    #             num_out=self.num_out,
    #             filters=self.filters,
    #             poolings=self.poolings,
    #             hidden_sizes=self.hidden_sizes,
    #             hidden_nl=self.hidden_nl,
    #             output_nl=self.output_nl)
    #     convnet.set_params(self.get_params())
    #     return convnet
