from model import LayerLab
from pylearn2.utils import wraps
from pylearn2.space import IndexSpace
from pylearn2.sandbox.rnn.space import SequenceDataSpace
from pylearn2.space import VectorSpace
from pylearn2.blocks import Block
from pylearn2.costs.mlp import Default
from pylearn2.utils import sharedX
import numpy as np
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


class WordEmbeding(LayerLab, Block):

    def __init__(self, dim, layer_name, input_space=None, use_affine=True,
                 seed=None, **kwargs):
        super(WordEmbeding, self).__init__(**kwargs)

        self.dim = dim
        self.layer_name = layer_name
        self.use_affine = use_affine

    def __call__(self, input):
        self.get_input_space().validate(input)
        return self.upward_pass(input)

    def _make_theano_rng(self, seed):
        # XXX: Need to handle random Stream management
        self.theano_rng = RandomStreams(seed)

    @wraps(LayerLab.set_input_space)
    def set_input_space(self, space):
        if not isinstance(space, IndexSpace):
            raise TypeError("WordEmbeding neeeds IndexSpace got: " + str(space))
        self.input_space = space

        rng = self.mlp.rng
        self.output_space = SequenceDataSpace(VectorSpace(dim=self.dim))
        self._make_theano_rng(max(rng.randint(2 ** 15), 1))

        self._Em = rng.normal(size=(self.input_space.max_labels, self.dim))
        self._W = rng.normal(size=(self.input_space.max_labels, self.dim))

        self._Em = sharedX(self._Em, name=self.layer_name + '_Em')
        self._W  = sharedX(self._W, name=self.layer_name + '_W')
        self._params = [self._Em, self._W]

        if self.use_affine:
            self._bias = np.zeros((self.input_space.max_labels,))
            self._bias = sharedX(self._bias, name=self.layer_name + '_b')
            self._params.append(self._bias)

    def upward_pass(self, state_below):
        """
        Support Pretrain Layer
        """
        self.input_space.validate(state_below)
        x = self._Em[state_below]  # if input is in
        x_shape = x.shape
        x = x.reshape((x_shape[1], x_shape[0], x_shape[2]))
        return x

    def fprop(self, state_below):
        if not isinstance(state_below, tuple):
            raise TypeError("State_below argument is not a tuple of tensor " +
                            "variable.\n" + str(state_below))
        self.input_space.validate(state_below)
        feature, center = state_below
        if self.use_affine:
            return T.nnet.sigmoid(
                T.sum(T.dot(self._Em[feature], self._W[center].T), axis=1) +
                self._bias[center])
        else:
            return self.non_affine(feature, center)

    def non_affine_fprop(self, feature, center):
            return T.nnet.sigmoid(T.sum(T.dot(self._Em[feature],
                                              self._W[center].T), axis=1))

    def _negative_target(self, center):
        """
        Return
        ------
        tensor vatiable : sample from word dict
        """
        return self.theano_rng.randn(high=self.input_space.max_labels,
                                     size=center.shape)

    def nce_loss(self, feature, center):
        """Build the graph for the NCE loss."""
        # cross-entropy(logits, labels)
        positive_output = self.fprob(feature, center)
        negative_target = self._negative_target(center)
        negative_output = self.fprob(feature, negative_target)
        nce_loss = T.nnet.binary_crossentropy(positive_output,
                                              T.ones_like(center)) + \
            T.nnet.binary_crossentropy(negative_output,
                    T.cast(T.eq(negative_target, positive_output), 'int32'))
        # NCE-loss is the sum of the true and noise (sampled words)
        # contributions, averaged over the batch.
        return nce_loss

    def cost(self, feature, target):
        return self.nce_loss(feature, target)

    @wraps(LayerLab.get_default_cost)
    def get_default_cost(self):
        return Default()


class SkipGramWordEmbeding(WordEmbeding):

    @wraps(WordEmbeding.set_input_space)
    def set_input_space_wrap(self, space):
        pass


class CBOWordEmbeding(WordEmbeding):

    @wraps(WordEmbeding.set_input_space)
    def set_space_space_dim(self, space):
        pass


if __name__ == "__main__":
    pass
