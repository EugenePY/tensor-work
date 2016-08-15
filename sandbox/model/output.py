import theano.tensor as T


class SoftMax(Layer):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = out_dim

    def __init_params(self):
        self.W = normal_init(input_dim, output_dim)
        self.bias = zeros_init((output_dim,))

    def output(self, input):
        return T.nnet.softmax(input)
