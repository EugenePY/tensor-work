import theano.tensor as T
from util.param_init import norm_weight, zero_init
from pylearn2.utils import wraps
from pylearn2.space import VectorSpace
from pylearn2.models.mlp import Linear
from pylearn2.sandbox.rnn.space import SequenceDataSpace
from model import LayerLab
from space import ContextSpace
from pylearn2.utils import sharedX
from pylearn2.linear.matrixmul import MatrixMul

class LinearAttention(Linear):

    @wraps(Linear.set_input_space)
    def set_input_space(self, space):
        self.input_space = space
        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)

        self.output_space = ContextSpace(
            dim=self.dim, num_annotation=self.input_space.num_annotation)

        rng = self.mlp.rng
        if self.irange is not None:
            assert self.istdev is None
            assert self.sparse_init is None
            W = rng.uniform(-self.irange,
                            self.irange,
                            (self.input_dim, self.dim)) * \
                (rng.uniform(0., 1., (self.input_dim, self.dim))
                 < self.include_prob)
        elif self.istdev is not None:
            assert self.sparse_init is None
            W = rng.randn(self.input_dim, self.dim) * self.istdev
        else:
            assert self.sparse_init is not None
            W = np.zeros((self.input_dim, self.dim))

            def mask_rejects(idx, i):
                if self.mask_weights is None:
                    return False
                return self.mask_weights[idx, i] == 0.

            for i in xrange(self.dim):
                assert self.sparse_init <= self.input_dim
                for j in xrange(self.sparse_init):
                    idx = rng.randint(0, self.input_dim)
                    while W[idx, i] != 0 or mask_rejects(idx, i):
                        idx = rng.randint(0, self.input_dim)
                    W[idx, i] = rng.randn()
            W *= self.sparse_stdev

        W = sharedX(W)
        W.name = self.layer_name + '_W'

        self.transformer = MatrixMul(W)

        W, = self.transformer.get_params()
        assert W.name is not None

        if self.mask_weights is not None:
            expected_shape = (self.input_dim, self.dim)
            if expected_shape != self.mask_weights.shape:
                raise ValueError("Expected mask with shape " +
                                 str(expected_shape) + " but got " +
                                 str(self.mask_weights.shape))
            self.mask = sharedX(self.mask_weights)



class HighWay(LayerLab):

    def __init__(self, dim, layer_name, activation='T.tanh'):
        super(HighWay, self).__init__()
        assert isinstance(activation, str)
        self._params = {'%s_%s'.format(__name__, param): None
                        for param in ['Wh', 'bh', 'Wt', 'bt']}
        self.layer_name = layer_name
        self.activation = activation
        self.dim = dim

    @wraps(LayerLab.set_input_space)
    def set_input_space(self, space):
        if not isinstance(space, SequenceDataSpace):
            raise TypeError("HighWay Layer currently do not support "
                            "None-SequenceDataSpace:(This Layer is currently "
                            "support Seq2Seq Model)")

        self.input_space = space
        self.output_space = SequenceDataSpace(VectorSpace(dim=self.dim))
        nin_dim = self.input_space.dim
        out_dim = self.output_space.dim
        self.Wh = norm_weight((nin_dim, out_dim))
        self.bh = zero_init((out_dim,))
        self.Wt = norm_weight((nin_dim, 1))
        self.bt = zero_init((1,))
        self._params.update(
            [(param_name, param) for param_name, param in
             zip(self._params.keys(), [self.Wh, self.bh, self.Wt, self.bt])])

    def fprop(self, input):
        t = T.nnet.sigmoid(T.dot(input, self.Wt) + self.bt)
        return t * eval(self.activation)(T.dot(input, self.Wh) + self.bh) + \
            (1 - t) * input


class ResidulWay(LayerLab):
    def __inti__(self, input_dim, outptut):
        pass


class FeedWay(LayerLab):
    def __init__(self, input_dim, out_dim, **kwargs):
        super(FeedWay, self).__init__(**kwargs)

    def __init_params(self):
        pass

    def fprop(self, input, activation):
        pass
