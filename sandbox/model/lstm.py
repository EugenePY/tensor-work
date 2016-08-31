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
from blocks.bricks.base import application, lazy
from blocks.bricks.parallel import Merge
from blocks.bricks.interfaces import Feedforward
from blocks.bricks.simple import Rectifier, Linear
from blocks.bricks.recurrent import recurrent, BaseRecurrent
from blocks.bricks import Random, Initializable, MLP

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


class GlimpseSensor(object):
    def __init__(self, channels, img_height, img_width, N):
        """A zoomable attention window for images.

        Parameters
        ----------
        channels : int
        img_heigt, img_width : int
            shape of the images
        N :
            $N \times N$ attention window size
        """
        self.channels = channels
        self.img_height = img_height
        self.img_width = img_width
        self.N = N

    def filterbank_matrices(self, center_y, center_x, delta, sigma):
        """Create a Fy and a Fx

        Parameters
        ----------
        center_y : T.vector (shape: batch_size)
        center_x : T.vector (shape: batch_size)
            Y and X center coordinates for the attention window
        delta : T.vector (shape: batch_size)
        sigma : T.vector (shape: batch_size)

        Returns
        -------
            FY : T.fvector (shape: )
            FX : T.fvector (shape: )
        """
        tol = 1e-4
        N = self.N

        rng = T.arange(N, dtype=floatX)-N/2.+0.5  # e.g.  [1.5, -0.5, 0.5, 1.5]

        muX = center_x.dimshuffle([0, 'x']) + delta.dimshuffle([0, 'x'])*rng
        muY = center_y.dimshuffle([0, 'x']) + delta.dimshuffle([0, 'x'])*rng

        a = tensor.arange(self.img_width, dtype=floatX)
        b = tensor.arange(self.img_height, dtype=floatX)

        FX = tensor.exp(-(a-muX.dimshuffle([0, 1, 'x']))**2 / 2. /
                        sigma.dimshuffle([0, 'x', 'x'])**2)
        FY = tensor.exp(-(b-muY.dimshuffle([0, 1, 'x']))**2 / 2. /
                        sigma.dimshuffle([0, 'x', 'x'])**2)
        FX = FX / (FX.sum(axis=-1).dimshuffle(0, 1, 'x') + tol)
        FY = FY / (FY.sum(axis=-1).dimshuffle(0, 1, 'x') + tol)

        return FY, FX

    def _batched_dot(self, A, B):
        C = A.dimshuffle([0, 1, 2, 'x']) * B.dimshuffle([0, 'x', 1, 2])
        return C.sum(axis=-2)

    def read(self, images, center_y, center_x, delta, sigma):
        """Extract a batch of attention windows from the given images.

        Parameters
        ----------
        images : :class:`~tensor.TensorVariable`
            Batch of images with shape (batch_size x img_size). Internally it
            will be reshaped to a (batch_size, img_height, img_width)-shaped
            stack of images.
        center_y : :class:`~tensor.TensorVariable`
            Center coordinates for the attention window.
            Expected shape: (batch_size,)
        center_x : :class:`~tensor.TensorVariable`
            Center coordinates for the attention window.
            Expected shape: (batch_size,)
        delta : :class:`~tensor.TensorVariable`
            Distance between extracted grid points.
            Expected shape: (batch_size,)
        sigma : :class:`~tensor.TensorVariable`
            Std. dev. for Gaussian readout kernel.
            Expected shape: (batch_size,)

        Returns
        -------
        windows : :class:`~tensor.TensorVariable`
            extracted windows of shape: (batch_size x N**2)
        """
        N = self.N
        channels = self.channels
        batch_size = images.shape[0]

        # Reshape input into proper 2d images
        I = images.reshape((batch_size*channels,
                            self.img_height, self.img_width))

        # Get separable filterbank
        FY, FX = self.filterbank_matrices(center_y, center_x, delta, sigma)

        FY = T.repeat(FY, channels, axis=0)
        FX = T.repeat(FX, channels, axis=0)

        # apply to the batch of images
        W = self._batched_dot(self._batched_dot(FY, I),
                              FX.transpose([0, 2, 1]))

        return W.reshape((batch_size, channels*N*N))

    def write(self, windows, center_y, center_x, delta, sigma):
        """Write a batch of windows into full sized images.

        Parameters
        ----------
        windows : :class:`~tensor.TensorVariable`
            Batch of images with shape (batch_size x N*N). Internally it
            will be reshaped to a (batch_size, N, N)-shaped
            stack of images.
        center_y : :class:`~tensor.TensorVariable`
            Center coordinates for the attention window.
            Expected shape: (batch_size,)
        center_x : :class:`~tensor.TensorVariable`
            Center coordinates for the attention window.
            Expected shape: (batch_size,)
        delta : :class:`~tensor.TensorVariable`
            Distance between extracted grid points.
            Expected shape: (batch_size,)
        sigma : :class:`~tensor.TensorVariable`
            Std. dev. for Gaussian readout kernel.
            Expected shape: (batch_size,)

        Returns
        -------
        images : :class:`~tensor.TensorVariable`
            extracted windows of shape: (batch_size x img_height*img_width)
        """
        N = self.N
        channels = self.channels
        batch_size = windows.shape[0]

        # Reshape input into proper 2d windows
        W = windows.reshape((batch_size*channels, N, N))

        # Get separable filterbank
        FY, FX = self.filterbank_matrices(center_y, center_x, delta, sigma)

        FY = T.repeat(FY, channels, axis=0)
        FX = T.repeat(FX, channels, axis=0)

        # apply...
        I = self._batched_dot(self._batched_dot(FY.transpose([0, 2, 1]), W), FX)

        return I.reshape((batch_size, channels*self.img_height*self.img_width))

    def nn2att(self, l):
        """Convert neural-net outputs to attention parameters

        Parameters
        ----------
        layer : :class:`~tensor.TensorVariable`
            A batch of neural net outputs with shape (batch_size x 5)

        Returns
        -------
        center_y : :class:`~tensor.TensorVariable`
        center_x : :class:`~tensor.TensorVariable`
        delta : :class:`~tensor.TensorVariable`
        sigma : :class:`~tensor.TensorVariable`
        gamma : :class:`~tensor.TensorVariable`
        """
        center_y = l[:, 0]
        center_x = l[:, 1]
        log_delta = l[:, 2]
        log_sigma = l[:, 3]
        log_gamma = l[:, 4]

        delta = T.exp(log_delta)
        sigma = T.exp(log_sigma/2.)
        gamma = T.exp(log_gamma).dimshuffle(0, 'x')

        # normalize coordinates
        center_x = (center_x+1.)/2. * self.img_width
        center_y = (center_y+1.)/2. * self.img_height
        delta = (max(self.img_width, self.img_height)-1) / (self.N-1) * delta

        return center_y, center_x, delta, sigma, gamma


class GlimpseNetwork(Initializable):
    def __init__(self, dims, n_channels, img_height, img_width, N,
                 activations=None, **kwargs):
        if not isinstance(dims, dict):
            raise TypeError("dims must be provided as a dict. ex "
                            "{0:[12, 23], 1: [12]}. Got " + str(dims))
        super(GlimpseNetwork, self).__init__(**kwargs)

        self.sensor = GlimpseSensor(channels=n_channels,
                                    img_height=img_height,
                                    img_width=img_width, N=N)

        self.glimpes_0 = Linear(input_dim=5, output_dim=dims.get(0)[0])
        self.glimpes_1 = Linear(input_dim=N*N*n_channels,
                                output_dim=dims.get(0)[1]),
        self.glimpes_out = Merge(input_names=['hidden_g0', 'hidden_g1'],
                                 input_dim=5+n_channels*(N**2),
                                 output_dim=dims.get(1)[0],
                                 prototype=Rectifier())
        self.children = [self.glimpes_0, self.glimpes_1, self.glimpes_out]
        self.output_dim = dims.get(1)[1]

    @application(inputs=['img', 'l_last'], outputs=['hidden_g'])
    def apply(self, img, l_last):
        """
        Params
        ------
        img: (batch_size, img_height, img_width, n_channels)
        center_x: (batch_size,)
        center_y: (batch_size,)
        ---

        Return
        ------
        h_g : (batch_size, output_dim)
        """
        l_last = self.sensor.nn2att(l_last)
        glimpes = self.sensor.read(img, *l_last)
        h0 = self.glimpes_0.apply(l_last)
        h0.name = 'hidden_g0'
        h1 = self.glimpes_1.apply(glimpes)
        h1.name = 'hidden_g1'
        h_g = self.glimpes_out.apply(h0, h1)
        return h_g


class LocationNetwork(Initializable, Random):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(LocationNetwork, self).__init__(**kwargs)
        self.prior_mean = 0.
        self.prior_log_sigma = 0.

        self.mean_transform = Linear(
                name=self.name+'_mean',
                input_dim=input_dim, output_dim=output_dim,
                weights_init=self.weights_init, biases_init=self.biases_init,
                use_bias=True)

        self.log_sigma_transform = Linear(
                name=self.name+'_log_sigma',
                input_dim=input_dim, output_dim=output_dim,
                weights_init=self.weights_init, biases_init=self.biases_init,
                use_bias=True)

        self.children = [self.mean_transform, self.log_sigma_transform]

    def get_dim(self, name):
        if name == 'input':
            return self.mean_transform.get_dim('input')
        elif name == 'output':
            return self.mean_transform.get_dim('output')
        else:
            raise ValueError

    @application(inputs=['x', 'u'], outputs=['z', 'kl_term'])
    def sample(self, x, u):
        """Return a samples and the corresponding KL term

        Parameters
        ----------
        x :

        Returns
        -------
        z : tensor.matrix
            Samples drawn from Q(z|x)
        kl : tensor.vector
            KL(Q(z|x) || P_z)

        """
        mean = self.mean_transform.apply(x)
        log_sigma = self.log_sigma_transform.apply(x)

        # Sample from mean-zeros std.-one Gaussian
        # u = self.theano_rng.normal(
        #            size=mean.shape,
        #            avg=0., std=1.)

        # ... and scale/translate samples
        z = mean + tensor.exp(log_sigma) * u  # mean-field

        # Calculate KL
        kl = (
            self.prior_log_sigma - log_sigma +
            0.5 * (
                tensor.exp(2 * log_sigma) + (mean - self.prior_mean) ** 2
                ) / tensor.exp(2 * self.prior_log_sigma) - 0.5
        ).sum(axis=-1)
        return z, kl

    # @application(inputs=['n_samples'])
    @application(inputs=['u'], outputs=['z_prior'])
    def sample_from_prior(self, u):
        """Sample z from the prior distribution P_z.

        Parameters
        ----------
        u : tensor.matrix
            gaussian random source

        Returns
        -------
        z : tensor.matrix
            samples

        """
        # z_dim = self.mean_transform.get_dim('output')

        # Sample from mean-zeros std.-one Gaussian
        # u = self.theano_rng.normal(
        #            size=(n_samples, z_dim),
        #            avg=0., std=1.)

        # ... and scale/translate samples
        z = self.prior_mean + tensor.exp(self.prior_log_sigma) * u  # mean-field
        # z.name("z_prior")

        return z


class RAM(BaseRecurrent, Random, Initializable):
    """
    Recurrent Attention Model (RAM)

    Paramerters
    -----------
    core : core type layer
    step_output : which space to output
    """

    def __init__(self, core, glimpes_network, location_network,
                 action_network, **kwargs):
        super(RAM, self).__init__(**kwargs)
        self.core = core  # projec to hidden state
        self.glimpes_network = glimpes_network  # sensor information
        self.action_network = action_network  # action network
        self.location_network = location_network
        self.children = [self.glimpes_network, self.core,
                         self.action_network]

    @recurrent(sequences=[],
               context=['img'], state=['l_last', 'state', 'cell'],
               outputs=['l', 'action', 'state_t', 'cell_t'])
    def apply(self, img, l_last, state, cell):

        hidden_g = self.glimpes_network.apply(img, l_last)
        l = self.location_network.apply(hidden_g)
        state_t, cell_t = self.core.apply(hidden_g, state, cell, iterate=False)
        action = self.action_network.apply(hidden_g)

        return l, action, state_t, cell_t


if __name__ == '__main__':
    test = np.random.normal(size=(5, 100, 200, 3))
    glim_net = GlimpseNetwork(dims={0: [12, 13], 1: [20]},
                              n_channels=3, img_height=100,
                              img_width=200, N=20)
    core =

