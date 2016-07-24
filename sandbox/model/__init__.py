import theano
import theano.tensor as T

import pylearn2.models import Model


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

    def init_params(self):
        return NotImplementedError('Abstract Layer:' + self.__prefix__)

    def fprob(self, state_below):
        return NotImplementedError('Abstract Layer:' + self.__prefix__)

    def activation(self, state_below):
        return NotImplementedError('Abstract Layer:' + self.__prefix__)


class MCMC(object):
    def __init__(self):
        pass

    def dist(self):
        pass

    def sample(self):
        pass
