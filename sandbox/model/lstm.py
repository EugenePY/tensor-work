""" Long-Short term Memory Module:
    --Version: 0.0
    --Last Editor: Eugene-Yuan Kow

An demostration about simple model using pylearn2

"""

import numpy as np
import theano
import theano.tensor as T

from pylearn2.space import VectorSpace
from pylearn2.sandbox.rnn.models.rnn import LSTM
from pylearn2.sandbox.rnn.space import (SequenceSpace, SequenceDataSpace)

from pylearn2.utils import wraps
from pylearn2.utils import sharedX

from model.attention import Attention

tensor = T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# Define FeatureSpace (Conveoltion ())
# Captiona Space (max sequence length, n_sample)
# Mask Space (max seq, n_sample) set([0, 1]),


class CondLSTM(LSTM):
    """
    Add condictional information of a LSTM : given some conditional information,
    for each time-steps.
    -------------------

    Model Parameters:
    =================
        Context :
    """
    def __init__(self, init_bias_cond=0., **kwargs):
        super(CondLSTM, self).__init__(**kwargs)
        self.rnn_friendly = True
        self.__dict__.update(locals())
        del self.self

    @wraps(LSTM.set_input_space)
    def set_input_space(self, input_space):
        assert isinstance(input_space, CompositeSpace)
        assert all([isinstance(a, b) for a, b in zip(input_space.components,
                                                     [SequenceSpace,
                                                      SequenceDataSpace])])
        self.input_space = input_space

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

        rng = self.mlp.rng

        if self.irange is None:
            raise ValueError("Recurrent layer requires an irange value in "
                             "order to initialize its weight matrices")

        input_dim = self.input_space.dim
        context_dim = self.input_space.components[-1].dim
        ###### Original Parameters
        # W is the input-to-hidden matrix
        W = rng.uniform(-self.irange, self.irange, (input_dim, self.dim * 4))

        # U is the hidden-to-hidden transition matrix
        U = rng.randn(self.dim, self.dim * 4)
        U, _ = scipy.linalg.qr(U)

        # b is the bias
        b = np.zeros((self.dim,))
        ####### Conditional Parameters (Project the context in the
              # Hidden Space of LSTM)
        Wc = rng.uniform(-self.irange, self.irange, (context_dim, self.dim * 4))

        Uc = rng.randn(self.dim, self.dim * 4)
        Uc, _ = scipy.linalg.qr(Uc)

        bc = np.zeros((self.dim,))



        self._params = [
            sharedX(W, name=(self.layer_name + '_W')),
            sharedX(U, name=(self.layer_name + '_U')),
            sharedX(b + self.init_bias,
                    name=(self.layer_name + '_b')),
            sharedX(Wc, name=(self.layer_name + '_Wc')),
            sharedX(Uc, namae=(self.layer_name + '_Uc')),
            sharedX(bc + self.init_bias_cond,
                    name=(self.layey_name + '_bc'))
        ]

    @wraps(LSTM.fprop)
    def fprop(self, state_below, return_all=False):
        assert len(state_below) in [2, 3]
        if len(state_below) == 3:
            state_below, mask, context_below = state_below
        else:
            state_below, context_below = state_below

        z0 = tensor.alloc(np.cast[config.floatX](0), state_below.shape[1],
                          self.dim * 2)

        z0 = tensor.unbroadcast(z0, 0)
        if self.dim == 1:
            z0 = tensor.unbroadcast(z0, 1)

        W, U, b, Wc, Uc, bc = self._params
        if self.weight_noise:
            W = self.add_noise(W)
            U = self.add_noise(U)
            Wc = self.add_noise(Wc)
            Uc = self.add_noise(Uc)

        state_below = tensor.dot(state_below, W) + b
        context_below = tensor.dot(context_below, Wc) + bc

        if mask is not None:
            (z, updates) = theano.scan(fn=self.fprop_step_mask,
                                       sequences=[state_below, mask],
                                       outputs_info=[z0],
                                       non_sequences=[context_below, Uc, U])
        else:
            (z, updates) = theano.scan(fn=self.fprop_step,
                                       sequences=[state_below],
                                       outputs_info=[z0],
                                       non_sequences=[context_below, Uc, U])

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

    def fprop_step_mask(self, state_below, mask,
                        state_before, context_below, Uc, U):
        """
        Scan function for case using masks

        Parameters
        ----------
        : todo
        state_below : TheanoTensor
        """

        g_on = state_below + context_below + \
            tensor.dot(state_before[:, :self.dim], U) + \
            tensor.dot(context_below[:, self.dim], Uc)
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

    def fprop_step(self, state_below, context_below, z, U, Uc):
        """
        Scan function for case without masks

        Parameters
        ----------
        : todo
        state_below : TheanoTensor
        """

        g_on = state_below + context_below + \
            tensor.dot(z[:, :self.dim], U) + \
            tensor.dot(z, Uc)
        i_on = tensor.nnet.sigmoid(g_on[:, :self.dim])
        f_on = tensor.nnet.sigmoid(g_on[:, self.dim:2*self.dim])
        o_on = tensor.nnet.sigmoid(g_on[:, 2*self.dim:3*self.dim])

        z = tensor.set_subtensor(z[:, self.dim:],
                                 f_on * z[:, self.dim:] +
                                 i_on * tensor.tanh(g_on[:, 3*self.dim:]))
        z = tensor.set_subtensor(z[:, :self.dim],
                                 o_on * tensor.tanh(z[:, self.dim:]))

        return z


class CondLSTMAttention(CondLSTM):
    """
    Param
    =====
        dim : int #hidden unit
        layer_name : str
        attention_mechenism : Attention model object
    """
    def __init__(self, attention, **kwargs):
        super(CondLSTMAttention, self).__init__(**kwargs)
        self.attention = attention

    def set_attention(self, attention):
        assert isinstance(attention, Attention)
        self.attention = attention

    def

    @wraps(CondLSTM.get_default_cost)
    def get_default_cost(self):
        pass
