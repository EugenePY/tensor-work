""" Long-Short term Memory Module:
    --Version: 0.0
    --Last Editor: Eugene-Yuan Kow

An demostration about simple model using pylearn2

"""
import functools
from pylearn2.models.model import Model

# from pylearn2.utils import wraps
from pylearn2.utils import shardedX
from pylearn2.blocks import Block, StackedBlocks

import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy

def options(dim_word=100,  # word vector dimensionality
          ctx_dim=512,  # context vector dimensionality
          dim=1000,  # the number of LSTM units
          attn_type='stochastic',  # [see section 4 from paper]
          n_layers_att=1,  # number of layers used to compute the attention weights
          n_layers_out=1,  # number of layers used to compute logit
          n_layers_lstm=1,  # number of lstm layers
          n_layers_init=1,  # number of layers to initialize LSTM at time 0
          lstm_encoder=False,  # if True, run bidirectional LSTM on input units
          prev2out=False,  # Feed previous word into logit
          ctx2out=False,  # Feed attention weighted ctx into logit
          alpha_entropy_c=0.002,  # hard attn param
          RL_sumCost=True,  # hard attn param
          semi_sampling_p=0.5,  # hard attn param
          temperature=1.,  # hard attn param
          patience=10,
          max_epochs=5000,
          dispFreq=100,
          decay_c=0.,  # weight decay coeff
          alpha_c=0.,  # doubly stochastic coeff
          lrate=0.01,  # used only for SGD
          selector=False,  # selector (see paper)
          n_words=10000,  # vocab size
          maxlen=100,  # maximum length of the description
          optimizer='rmsprop',
          batch_size = 16,
          valid_batch_size = 16,
          saveto='model.npz',  # relative path of saved model file
          validFreq=1000,
          saveFreq=1000,  # save the parameters after every saveFreq updates
          sampleFreq=100,  # generate some samples after every sampleFreq updates
          dataset='flickr8k',
          dictionary=None,  # word dictionary
          use_dropout=False,  # setting this true turns on dropout at various points
          use_dropout_lstm=False,  # dropout on lstm gates
          reload_=False,
          save_per_epoch=False): # this saves down the model every epoch
    return locals().copy()


# some utilities
def _p(prefix, name):
    return '%s_%s' % (prefix, name)

layers = {'ff': ('fflayer'),
          'lstm': ('lstm_layer'),
          'lstm_cond': 'lstm_cond_layer'),
          }

def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))

def ortho_weight(ndim):
    """
    Random orthogonal weights

    Used by norm_weights(below), in which case, we
    are ensuring that the rows are orthogonal
    (i.e W = U \Sigma V, U has the same
    # of rows, V has the same # of cols)
    """
    W = numpy.random.randn(ndim, ndim)
    u, _, _ = numpy.linalg.svd(W)
    return u.astype('float32')


def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    """
    Random weights drawn from a Gaussian
    """
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')


# some useful shorthands
def tanh(x):
    return tensor.tanh(x)


def rectifier(x):
    return tensor.maximum(0., x)


def linear(x):
    return x

# How to use pylearn2.models.Model
# Overider the method which you want to implement first
class Layer(Model):
    """ Abstract Class for Layer """
    __prefix__ = 'asbtract'
    params = {}

    def fprob(self, state_below):
        return NotImplementedError('Abstract Layer:' + self.__prefix__)

    def activation(self, state_below):
        return NotImplementedError('Abstract Layer:' + self.__prefix__)


class DropOut(Layer):
    """ DropOut Layer
    """
    def __init__(self, selelctor=False, rng=None):
        if rng is None:
            rng = RandomStreams(2**123)

        self.rng = rng

    def lstm_dropout(self, one_timestep):
        assert hasattr(one_timestep, '__call__')
        assert self.__prefix__ == 'lstm'

        one_timestep.local()

    def dropout(self, state_below, use_noise):
        proj = T.switch(use_noise, state_before *
                        self.rng.binomial(state_before.shape,p=0.5, n=1,
                                          dtype=state_before.dtype),
                        state_before *  0.5)
        return proj

    def fprob_dropout(self):
        pass

class LSTMDropOut(LSTM, DropOut):
    pass


class LSTM(Layer):
    """ version: SAT """

    def __init__(self, state_below, nin, dim, **argv, **kwargs):
        """
        input:
            state_below: last_layer's output
        """
        super(LSTM, self).__init__(**argv, **kwargs)

        # input info do not initiate here
        self.dim = dim
        self.nin = nin
        self.n_steps = state_below.shape[0]
        self.__prefix__ = "lstm"

        if state_below.ndim == 3:
            self.n_samples = state_below.shape[1]
            self.init_state = tensor.alloc(0., self.n_samples, dim)
            self.init_memory = tensor.alloc(0., self.n_samples, dim)
        # during sampling
        else:
            self.n_samples = 1
            self.init_state = tensor.alloc(0., dim)
            self.init_memory = tensor.alloc(0., dim)


    def __slice(self, x, n, dim):
        if x.ndim == 3:
            return x[:, :, n*dim:(n+1)*dim]
        elif x.ndim == 2:
            return x[:, n*dim:(n+1)*dim]
        return x[n*dim:(n+1)*dim]

    def one_timestep(self, mask, x_, h_, c_): # Why Mask ?????
        """ corresponding to the def _step() """
        preact = tensor.dot(h_, self.U)
        preact += x_

        i = tensor.nnet.sigmoid(self.__slice(preact, 0, self.dim))
        f = tensor.nnet.sigmoid(self.__slice(preact, 1, self.dim))
        o = tensor.nnet.sigmoid(self.__slice(preact, 2, self.dim))
        c = tensor.tanh(self.__slice(preact, 3, dim))

        c = f * c_ + i * c
        h = o * tensor.tanh(c)

        return h, c, i, f, o, preact

    def param_init_lstm(self, params={}): # TODO: Make this less shitty...
        """
        Stack the weight matricies for all the gates
        for much cleaner code and slightly faster dot-prods
        """
        self.params = params
        # input weights
        if self.params.get(_p(self.__prefix__, 'W'), None):
            W = numpy.concatenate([norm_weight(self.nin, self.dim),
                                   norm_weight(self.nin, self.dim),
                                   norm_weight(self.nin, self.dim),
                                   norm_weight(self.nin, self.dim)], axis=1)

        if self.params.get(_p(self.__prefix__, 'U'), None):
            U = numpy.concatenate([ortho_weight(self.dim),
                                   ortho_weight(self.dim),
                                   ortho_weight(self.dim),
                                   ortho_weight(self.dim)], axis=1)
        if params.get(_p(self.__prefix__, 'b'), None):
            b = numpy.zeros((4 * self.dim,)).astype('float32')


        # shared the variables
        self.W, self.U, self.b = map(
            lambda x, y: shardedX(x, name=_p(self.__prefix__, y)),
            zip([W, U, b], ['W', 'U', 'b']))

        self.params[_p(self.__prefix__, 'b')] = self.b
        self.params[_p(self.__prefix__, 'U')] = self.U
        self.params[_p(self.__prefix__, 'W')] = self.W

        return self

    def fprop(self, state_below, mask=None):
        """ forward probagation of LSTM
            input: (all are tensor variables)
        """
        if mask == None:
            mask = tensor.alloc(1., state_below.shape[0], 1)

        state_below = tensor.dot(state_below, self.W) + self.b

        rval, updates = theano.scan(self.one_timestep,
                                sequences=[mask, state_below],
                                outputs_info=[self.init_state, self.init_memory,
                                              None, None, None, None],
                                name=_p(prefix, 'layers'),
                                n_steps=nsteps, profile=False)

        return ravel, updates

    def set_input_space(self):
        pass


class LSTM_with_Attention(LSTM, Block):
    def __init__(self, input, context, prefix='lstm_cond',
                 attn_type='stochastic', **argv, **kargv):
        """
        input: the input sequence ()
        context: Z_{t} (alpha, )

        """
        super(LSTM_with_Attention, self).__init__()
        self.attn_type = attn_type
        self.Wc_att = None
        self.b_att = None
        self.W = None
        self.U = None

    def initial_params(self):
        pass

    def one_timestep(self):
        pass

    def attn_type(self, attn_type='stochastic'):
        def attention_decorator(func):
            @functools.wraps(func)
            def func_wrapper():
                pass

    def attention_mechenism(self):
        """
        """
        pass


class Embeding(Model, Block):
    pass


class WordEmbededing(Embeding):
    pass


class ContextEmbededfing(Embededing):
    pass


def check_the_config(layers):
    """ Check the layers are input & output coherent with each others"""
    for layer in layers:
        pass


class SAT(Model, Block, StackBlocks):
    """ Show, Attend and Tell:
        Neural Image Caption Generation with Visual Attention """

    def __init__(self, layers):
        # Checking wheather the inputs is valided.
        pass

    def _update_layer_input_spaces(self):
        """
        Tells each layer what its input space should be.

        Notes
        -----
        This usually resets the layer's parameters!
        """
        layers = self.layers
        try:
            layers[0].set_input_space(self.get_input_space())
        except BadInputSpaceError as e:
            raise TypeError("Layer 0 (" + str(layers[0]) + " of type " +
                            str(type(layers[0])) +
                            ") does not support the SAT's " +
                            "specified input space (" +
                            str(self.get_input_space()) +
                            " of type " + str(type(self.get_input_space())) +
                            "). Original exception: " + str(e))
        for i in xrange(1, len(layers)):
            layers[i].set_input_space(layers[i - 1].get_output_space())

    def build_model(self):
        pass


if __name__ == "__main__":
    pass
