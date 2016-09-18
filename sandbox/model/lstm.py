""" Long-Short term Memory Module:
    --Version: 0.0
    --Last Editor: Eugene-Yuan Kow
"""

import scipy
import theano
import theano.tensor as T
from pylearn2.models.mlp import MLP

from pylearn2.space import VectorSpace
from pylearn2.sandbox.rnn.models.rnn import LSTM
from pylearn2.space import CompositeSpace
from pylearn2.sandbox.rnn.space import (SequenceSpace,
                                        SequenceDataSpace)
from pylearn2.utils import wraps
from pylearn2.utils import sharedX

import numpy as np

# Blocks
floatX = theano.config.floatX

# from model.attention import Attention
from space import ContextSpace
from model.mlp_hook import AttentionWrapper
from model import LayerLab
tensor = T
# from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


# Define FeatureSpace (Conveoltion ())
# Captiona Space (max sequence length, n_sample)
# Mask Space (max seq, n_sample) set([0, 1]),
class ContextLSTM(LSTM):
    __metaclass__ = AttentionWrapper


class ConditionLSTM(LSTM):
    """
    Add condictional information of a LSTM : given some conditional information,
    for each time-steps.
    -------------------

    Model Parameters:
    =================
        Context :
        Attention: optional a Attention object
    """
    def __init__(self, init_bias_cond=0., attention=None, **kwargs):
        super(ConditionLSTM, self).__init__(**kwargs)
        self.rnn_friendly = True
        self.__dict__.update(locals())
        del self.self

        if attention is not None:
            self.attention = attention
            self.attention.set_lstm(self)
            self.use_attention = True
        else:
            self.use_attention = False

    def _setup_attention(self):
        assert self.use_attention
        attention_space = CompositeSpace([self.context_space,
                                          VectorSpace(dim=self.dim)])
        self.attention.set_input_space(attention_space)

    @wraps(LSTM.set_input_space)
    def set_input_space(self, input_space):
        assert isinstance(input_space, CompositeSpace)
        if not all([isinstance(a, b) for a, b in zip(input_space.components,
                                                     [SequenceSpace,
                                                      ContextSpace])]):
            raise TypeError("Contaional LSTM only takes :SequenceSpace,"
                            "ContextSpace. Got " + str(input_space))
        self.input_space = input_space
        self.context_space = input_space.components[-1]

        if self.indices is not None:
            if len(self.indices) > 1:
                raise ValueError("Only indices = [-1] is supported right now")
                self.output_space = CompositeSpace(
                    [VectorSpace(dim=self.dim) for _
                     in range(len(self.indices))]
                )
            else:
                assert self.indices == [-1], "Only indices = [-1] works now"
                self.output_space = VectorSpace(dim=self.dim)
        else:
            if isinstance(self.input_space, SequenceSpace):
                self.output_space = SequenceSpace(VectorSpace(dim=self.dim))
            elif isinstance(self.input_space, SequenceDataSpace):
                self.output_space =\
                    SequenceDataSpace(VectorSpace(dim=self.dim))

        if self.use_attention:
            self.attention.set_mlp(self.mlp)
            self._setup_attention()

        rng = self.mlp.rng

        if self.irange is None:
            raise ValueError("Recurrent layer requires an irange value in "
                             "order to initialize its weight matrices")

        input_dim = self.input_space.components[0].dim
        context_dim = self.context_space.dim
        # Original Parameters
        # W is the input-to-hidden matrix
        W = rng.uniform(-self.irange, self.irange, (input_dim, self.dim * 4))

        # U is the hidden-to-hidden transition matrix
        U = np.zeros((self.dim, self.dim * 4))
        for i in xrange(4):
            u = rng.randn(self.dim, self.dim)
            U[:, i*self.dim:(i+1)*self.dim], _ = scipy.linalg.qr(u)

        # b is the bias
        b = np.zeros((self.dim * 4,))

        # Conditional Parameters (Project the context in the
        # Hidden Space of LSTM)
        Wc = rng.uniform(-self.irange, self.irange, (context_dim, self.dim * 4))
        # Uc = rng.randn(self.dim, self.dim * 4)
        # Uc, _ = scipy.linalg.qr(Uc)
        # bc = np.zeros((self.dim,))

        self._params = [
            sharedX(W, name=(self.layer_name + '_W')),
            sharedX(U, name=(self.layer_name + '_U')),
            sharedX(b + self.init_bias, name=(self.layer_name + '_b')),
            sharedX(Wc, name=(self.layer_name + '_Wc'))
            # sharedX(Uc, name=(self.layer_name + '_Uc')),
            # sharedX(bc + self.init_bias_cond, name=(self.layer_name + '_bc'))
        ]

    @wraps(LSTM.fprop)
    def fprop(self, state_below, z0=None, return_all=False):
        assert len(state_below) in [2, 3]
        if len(state_below) == 3:
            state_below, mask, context = state_below
        else:
            state_below, context = state_below
            mask = None

        # XXX making this else where
        # input checking
        self.context_space = self.input_space.components[1]
        self.context_space.validate(context)

        if z0 is None:
            # Init memory and state
            z0 = tensor.alloc(np.cast[theano.config.floatX](0),
                              state_below.shape[1],
                              self.dim * 2)
        else:
            z0 = z0

        z0 = tensor.unbroadcast(z0, 0)
        if self.dim == 1:
            z0 = tensor.unbroadcast(z0, 1)

        W, U, b, Wc = self._params
        if self.weight_noise:
            W = self.add_noise(W)
            U = self.add_noise(U)
            Wc = self.add_noise(Wc)

        state_below = tensor.dot(state_below, W) + b

        if self.use_attention:
            pctx = self.attention._project(context)

        if mask is not None:
            if self.use_attention:
                (z, updates) = theano.scan(fn=self.fprop_step_attention_mask,
                                           sequences=[state_below, mask],
                                           outputs_info=[z0],
                                           non_sequences=[pctx, Wc, U])
            else:

                (z, updates) = theano.scan(fn=self.fprop_step_mask,
                                           sequences=[state_below, mask],
                                           outputs_info=[z0],
                                           non_sequences=[context, Wc, U])
        else:
            if self.use_attention:
                (z, updates) = theano.scan(fn=self.fprop_step_attention,
                                           sequences=[state_below],
                                           outputs_info=[z0],
                                           non_sequences=[pctx, Wc, U])
            else:
                (z, updates) = theano.scan(fn=self.fprop_step,
                                           sequences=[state_below],
                                           outputs_info=[z0],
                                           non_sequences=[context, Wc, U])

            self._scan_updates.update(updates)

        if return_all:
            return z

        if self.indices is not None:
            if len(self.indices) > 1:
                return [z[i, :, :self.dim] for i in self.indices]
            else:
                return z[self.indices[0], :, :self.dim]
        else:
            if mask is not None:
                return (z[:, :, :self.dim], mask)
            else:
                return z[:, :, :self.dim]

    def fprop_step_mask(self, state_below, mask, state_before, context, Wc, U):
        """
        Scan function for case using masks

        Parameters
        ----------
        : todo
        state_below : TheanoTensor
        state_before : TheanoTensor :(batch_size, dim * 2)
            [:dim] hidden state
            [dim:] memory
        """
        if self.use_attention:
            context = context.sum(1)
        else:
            context = context.mean(1)

        g_on = state_below + \
            tensor.dot(state_before[:, :self.dim], U) + \
            tensor.dot(context, Wc)

        i_on = tensor.nnet.sigmoid(g_on[:, :self.dim])
        f_on = tensor.nnet.sigmoid(g_on[:, self.dim:2*self.dim])
        o_on = tensor.nnet.sigmoid(g_on[:, 2*self.dim:3*self.dim])

        z = tensor.set_subtensor(state_before[:, self.dim:],
                                 f_on * state_before[:, self.dim:] +
                                 i_on * tensor.tanh(g_on[:, 3*self.dim:]))
        z = tensor.set_subtensor(z[:, :self.dim],
                                 o_on * tensor.tanh(z[:, self.dim:]))

        # Only update the state for non-masked data, otherwise
        # just carry on the previous state until the end
        z = mask[:, None] * z + (1 - mask[:, None]) * state_before

        return z

    def fprop_step_attention_mask(self, state_below, mask, state_before,
                                  pctx, Uc, U):

        ctx = self.attention.alpha_sample(state_before[:, :self.dim], pctx)
        z = self.fprop_step_mask(state_below, mask, state_below, ctx)
        return z

    def fprop_step(self, state_below, context, z, U, Uc):
        """
        Scan function for case without masks

        Parameters
        ----------
        : todo
        state_below : TheanoTensor
        """

        g_on = state_below + context + \
            tensor.dot(z[:, :self.dim], U) + \
            tensor.dot(z[:, :self.dim], Uc)
        i_on = tensor.nnet.sigmoid(g_on[:, :self.dim])
        f_on = tensor.nnet.sigmoid(g_on[:, self.dim:2*self.dim])
        o_on = tensor.nnet.sigmoid(g_on[:, 2*self.dim:3*self.dim])
        # update hidden state
        z = tensor.set_subtensor(z[:, self.dim:],
                                 f_on * z[:, self.dim:] +
                                 i_on * tensor.tanh(g_on[:, 3*self.dim:]))
        # update memory
        z = tensor.set_subtensor(z[:, :self.dim],
                                 o_on * tensor.tanh(z[:, self.dim:]))

        return z



if __name__ == '__main__':
    test = np.random.normal(size=(5, 100, 200, 3))
    glim_net = GlimpseNetwork(dims={0: [12, 13], 1: [20]},
                              n_channels=3, img_height=100,
                              img_width=200, N=20)
