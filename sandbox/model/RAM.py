import theano
import theano.tensor as T
import numpy as np

from blocks.bricks.base import application
from blocks.bricks.simple import Linear, Softmax
from blocks.bricks.recurrent import recurrent, LSTM, BaseRecurrent
from blocks.bricks import Random, Initializable
from blocks.initialization import IsotropicGaussian, Constant

floatX = theano.config.floatX


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

    def get_dim(self, name):
        if name == 'W':
            return self.channel * self.N**2
        elif name == 'img':
            return self.channels * self.img_width * self.img_height

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

        a = T.arange(self.img_width, dtype=floatX)
        b = T.arange(self.img_height, dtype=floatX)

        FX = T.exp(-(a-muX.dimshuffle([0, 1, 'x']))**2 / 2. /
                   sigma.dimshuffle([0, 'x', 'x'])**2)
        FY = T.exp(-(b-muY.dimshuffle([0, 1, 'x']))**2 / 2. /
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

        delta = T.exp(log_delta)
        sigma = T.exp(log_sigma/2.)

        # normalize coordinates
        center_x = (center_x+1.)/2. * self.img_width
        center_y = (center_y+1.)/2. * self.img_height
        delta = (max(self.img_width, self.img_height)-1) / (self.N-1) * delta

        return center_y, center_x, delta, sigma


class GlimpseNetwork(Initializable):
    """
    GlimpseSensor & Linear + Rectifier
    ----------------------------------

    apply:
        input_shape (batch_size, n_channels * img_width * img_height)
        output_dim (batch_size, dim)

    """
    def __init__(self, loc_emb, dim, n_channels, img_height, img_width, N,
                 activations=None, **kwargs):

        super(GlimpseNetwork, self).__init__(**kwargs)
        self.loc_emb = loc_emb
        self.sensor = GlimpseSensor(channels=n_channels,
                                    img_height=img_height,
                                    img_width=img_width, N=N)

        self.glimpes_0 = Linear(input_dim=loc_emb, output_dim=4,
                                name=self.name + '_glimp_0',
                                weights_init=self.weights_init,
                                biases_init=self.biases_init)

        self.glimpes_1 = Linear(input_dim=N*N*n_channels,
                                output_dim=dim, name=self.name + '_glimp_1',
                                weights_init=self.weights_init,
                                biases_init=self.biases_init)

        self.children = [self.glimpes_0, self.glimpes_1]  # self.glimpes_out]
        self.output_dim = dim

    def get_dim(self, name):
        if name == 'img':
            return self.sensor.get_dim('img')
        elif name == 'l_last':
            return self.loc_emb
        else:
            raise ValueError

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
        h0 = self.glimpes_0.apply(l_last)
        l_unpack = self.sensor.nn2att(h0)
        glimpes = self.sensor.read(img, *l_unpack)
        h1 = self.glimpes_1.apply(glimpes)
        hidden_g = T.nnet.relu(h1)
        return hidden_g


class LocationNetwork(Initializable, Random):
    def __init__(self, input_dim, loc_emb, sensor, **kwargs):
        super(LocationNetwork, self).__init__(**kwargs)
        # self.linear = Linear()
        self.loc_emb = loc_emb
        self.transform = Linear(
                input_dim=input_dim, output_dim=loc_emb,
                weights_init=self.weights_init,
                biases_init=self.biases_init)

        self.sensor = sensor
        self.children = [self.transform]

    def get_dim(self, name):
        if name == 'hidden_g':
            return self.transform.get_dim('inputs')
        elif name == 'l':
            return self.transform.get_dim('outputs')
        else:
            raise ValueError

    @application(inputs=['hidden_g'], outputs=['l'])
    def apply(self, hidden_g):
        return self.transform.apply(hidden_g)


class CoreNetwork(Initializable):
    def __init__(self, input_dim, dim, **kwargs):
        super(CoreNetwork, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.dim = dim
        self.lstm = LSTM(dim=dim, name=self.name + '_lstm',
                         weights_init=self.weights_init,
                         biases_init=self.biases_init)

        self.proj = Linear(input_dim=input_dim, output_dim=dim*4,
                           name=self.name + '_proj',
                           weights_init=self.weights_init,
                           biases_init=self.biases_init)
        self.children = [self.lstm, self.proj]

    def get_dim(self, name):
        if name == 'inputs':
            return self.input_dim
        elif name in ['state', 'cell']:
            return self.dim
        else:
            raise ValueError

    @application
    def apply(self, inputs, state, cell):
        state, cell = self.lstm.apply(self.proj.apply(inputs), state, cell,
                                      iterate=False)
        return state, cell


class ActionNetwork(Initializable):
    def __init__(self, input_dim, n_classes, **kwargs):
        super(ActionNetwork, self).__init__(**kwargs)
        self.transform = Linear(input_dim=input_dim,
                                output_dim=n_classes, **kwargs)
        self.out = Softmax()

        self.children = [self.transform, self.out]

    def get_dim(self, name):
        if name == 'hidden_g':
            return self.transform.get_dim('inputs')
        else:
            raise ValueError

    @application(inputs=['hidden_g'], outputs=['action'])
    def apply(self, hidden_g):
        return self.out.apply(self.transform.apply(hidden_g))


class RAM(BaseRecurrent, Initializable):
    """
    Recurrent Attention Model (RAM)

    Paramerters
    -----------
    core : core type layer
    step_output : which space to output
    """

    def __init__(self, core, glimpes_network, location_network,
                 action_network, n_steps, **kwargs):

        super(RAM, self).__init__(**kwargs)
        self.core = core  # projec to hidden state
        self.glimpes_network = glimpes_network  # sensor information
        self.action_network = action_network  # action network
        self.location_network = location_network
        self.n_steps = n_steps

        self.children = [self.glimpes_network, self.core,
                         self.location_network,
                         self.action_network]

    def get_dim(self, name):
        if name in ['img', 'null_var']:
            return self.glimpes_network.get_dim('img')
        elif name == 'l_last':
            return self.glimpes_network.loc_emb
        elif name == 'action':
            return self.action_network.output_dim
        elif name == 'state':
            return self.core.get_dim('state')
        elif name == 'cell':
            return self.core.get_dim('cell')
        else:
            return super(RAM, self).get_dim(name)

    @recurrent(sequences=['null_var'],
               contexts=['img'], states=['l_last', 'state', 'cell'],
               outputs=['l_last', 'action', 'state', 'cell'])
    def apply(self, null_var, img, l_last, state, cell):
        img = img + null_var
        hidden_g = self.glimpes_network.apply(img, l_last)
        state, cell = self.core.apply(hidden_g, state, cell)
        action = self.action_network.apply(state)
        l = self.location_network.apply(state)
        return l, action, state, cell

    @application(inputs=['img'],
                 outputs=['l', 'action', 'state', 'cell'])
    def out(self, img):
        null_var = T.zeros((self.n_steps, img.shape[0], img.shape[1]))
        null_var.name = 'null_var'
        l, action, state, cell = self.apply(null_var, img)
        return l, action, state, cell

    @recurrent(sequences=[])
    def sample_loc(self, img):
        pass

if __name__ == '__main__':
    from blocks.graph import ComputationGraph

    test = (np.random.normal(size=(5, 100 * 200 * 3)).astype(floatX),
            np.random.normal(size=(5, 5)).astype(floatX),
            np.random.normal(size=(5, 30)).astype(floatX),
            np.random.normal(size=(5, 30)).astype(floatX))

    inits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }

    glim_net = GlimpseNetwork(dim=12,
                              n_channels=3, img_height=100,
                              img_width=200, N=20, name='glimpes_net',
                              **inits)
    core = CoreNetwork(input_dim=12, dim=30, name='core_net', **inits)
    loc_net = LocationNetwork(30, glim_net.sensor, name='loc_net', **inits)
    # Test loc net
    action = ActionNetwork(30, 30, **inits)
    ram = RAM(core, glim_net, loc_net, action, 30, name='RAM', **inits)
    ram.initialize()

    img = T.matrix('img')
    l = T.matrix('l_last')
    state = T.matrix('state')
    cell = T.matrix('cell')

    # Tests
    h_g = glim_net.apply(img, l)
    rval = h_g.eval({img: test[0], l: test[1]})
    assert rval.shape == (5, 12)
    print str(glim_net) + ' Pass ...'
    os = core.apply(h_g, state, cell)
    rvals = [o.eval({img: test[0], l: test[1], state:test[2], cell:test[3]})
             for o in os]

    assert all([rvali.shape == (5, 30) for rvali in rvals])
    print str(core) + ' Pass ...'
    # rval = action.applys()
    fn = ram.out(img)
    cg = ComputationGraph(fn)
    out = theano.function([img], fn,
                          allow_input_downcast=True)
    res = out(test[0])
    print [rvalj.shape for rvalj in res]
    print res[1][0]
