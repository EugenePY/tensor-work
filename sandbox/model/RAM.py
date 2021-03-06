import theano
import theano.tensor as T
import theano.tensor.nnet.neighbours
import numpy as np

from blocks.bricks.base import application  # , Brick
from blocks.bricks.simple import Linear, Softmax, Logistic
from blocks.bricks.recurrent import recurrent, LSTM, BaseRecurrent
from blocks.bricks import Random, Initializable
from blocks.initialization import IsotropicGaussian, Constant
from sample_location import img_ratangle
floatX = theano.config.floatX


def hard_tanh(x):
    return T.clip(x, -1, 1)
    # return T.tanh(x)


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
        self.emb_dim = 4

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


class GlimpseSensorBeta(GlimpseSensor):
    """
    Pure Location-based Sensor,
        --- Continuous location
    """
    def __init__(self, boarder=[[-1., 1.], [-1., 1.]], stride_size=None,
                 **kwargs):
        super(GlimpseSensorBeta, self).__init__(**kwargs)
        self.boarder = boarder
        if stride_size is None:
            stride_size = (self.N, self.N)  # None overlap stride
        self.stride_size = stride_size
        self.emb_dim = 2
        self.step_x, self.step_y = self.stride_size
        self.total_step_x = (self.img_width - self.N) // self.step_x + 1
        self.total_step_y = (self.img_height - self.N) // self.step_y + 1

        if any([self.total_step_x <= 0, self.total_step_y <= 0]):
            raise ValueError('stride size to big')

        self.total_step = self.total_step_x * self.total_step_y

    def get_dim(self, name):
        if name == 'glimpse':
            return (self.N) ** 2 * self.channels
        else:
            raise ValueError

    def map_float_to_index(self, center_x, center_y):
        x_board, y_board = self.boarder

        loc_x = T.cast((self.total_step_x - 1) *
                       (center_x - x_board[0]) // 2, 'int32')
        loc_y = T.cast((self.total_step_y - 1) *
                       (center_y - y_board[0]) // 2, 'int32')
        idx = self.total_step_y * loc_x + loc_y
        return idx

    # @application(inputs=['img', 'center_x', 'center_y'], outputs=['glimpes'])
    def read(self, img, center_x, center_y):
        idx = self.map_float_to_index(center_x, center_y)
        batch_size = img.shape[0]
        img_4d = img.reshape((batch_size, self.channels, self.img_height,
                              self.img_width))
        neighbours = T.nnet.neighbours.images2neibs(img_4d,
                                                    neib_shape=(self.N, self.N),
                                                    neib_step=self.stride_size)
        # (batch_size * dim * step_x * step_y)
        neighbours_new = T.reshape(neighbours, (batch_size,
                                   self.channels, self.total_step,
                                   self.N * self.N), ndim=4)

        def select_step(sub_neibor, id):
            return sub_neibor[:, id]

        nei, _ = theano.map(select_step, sequences=[neighbours_new, idx])
        return nei.reshape((batch_size, self.channels * self.N * self.N))

    def att_mark(self, img, center_x, center_y):
        """
        This method is for monitoring the training behavior.
        """
        # action = action + 1  # making color is different from others (0, 1)

        idx = self.map_float_to_index(center_x, center_y)
        batch_size = img.shape[0]
        img_4d = img.reshape((batch_size, self.channels, self.img_height,
                              self.img_width))
        retagle_idx_x, retagle_idx_y = img_ratangle(
            (self.N, self.N))
        neighbours = T.nnet.neighbours.images2neibs(img_4d,
                                                    neib_shape=(self.N, self.N),
                                                    neib_step=self.stride_size)
        # (batch_size * dim * step_x * step_y)
        neighbours_new = T.reshape(neighbours, (batch_size,
                                   self.channels, self.total_step,
                                   self.N, self.N), ndim=5)
        if self.channels == 1:
            neighbours_new = T.repeat(neighbours_new, 3, axis=1)

        def select_step(sub_neibor, id):
            sub_neibor = T.set_subtensor(
                sub_neibor[0, id, retagle_idx_x, retagle_idx_y], 255)
            sub_neibor_new = T.set_subtensor(
                sub_neibor[1:, id, retagle_idx_x, retagle_idx_y], 0)
            return sub_neibor_new

        nei, _ = theano.scan(select_step,
                             sequences=[neighbours_new, idx],
                             outputs_info=[None])
        if self.channels == 1:
            nei = nei.reshape((batch_size * 3 * self.total_step,
                              self.N * self.N))
        else:
            nei = nei.reshape((batch_size * 3 * self.total_step,
                              self.N * self.N))

        if self.stride_size != (self.N, self.N):
            raise ValueError('Current do not support none default stride_size.'
                             ' Got %s'.format(self.stride_size,))
        imgs = T.nnet.neighbours.neibs2images(
            nei, (self.N, self.N), (self.img_height, self.img_width))
        if self.channels == 1:
            out = imgs.reshape((batch_size, 3, self.img_height,
                                self.img_width))
        return out

    def write(self, w, center_x, center_y):
        x_board, y_board = self.boarder
        step_x, step_y = self.stride_size
        # this should be interger
        total_step_x = (self.img_width - self.N) // step_x + 1
        total_step_y = (self.img_height - self.N) // step_y + 1

        idx = self.map_float_to_index(center_x, center_y)
        batch_size = w.shape[0]

        w_buffer = T.zeros((batch_size * self.channels,
                            total_step_x * total_step_y,
                            self.N, self.N))

        w_buffer = T.set_subtensor(w_buffer[:, idx], w.reshape((batch_size,
                                                                self.channels,
                                                                self.N,
                                                                self.N)))
        imgs = T.nnet.neighbours.neibs2images(
            w_buffer, (self.N, self.N), (self.img_height, self.img_width))
        return imgs

    def nn2att(self, l):
        center_x = l[:, 0]
        center_y = l[:, 1]

        # center_x = T.tanh(center_x_emb)
        # center_y = T.tanh(center_y_emb)
        return center_x, center_y


class RetinaGlimpse(object):
    """
    Location-based Retina Sensor,
        --- Continuous location, this can do more complex glimpes operation
        for each (x, y) we not only do chop but also do pooling.

    """
    __axis_order = ('b', 'r', 'c', 0, 1)
    # batch, retina, channel, img_heigh, img_width

    def __init__(self, img_width, img_height, channels,
                 n_retina=3, retina_strides=(2, 2), radius=3,
                 boarder=[[-1., 1.], [-1., 1.]], include_center=False):

        self.include_center = include_center
        self.boarder = boarder
        self.img_width = img_width
        self.img_height = img_height
        self.channels = channels
        self.n_retina = n_retina
        self.retina_strides = retina_strides
        self.emb_dim = 2
        self.radius = radius
        # XXX: Need to check the retina stride if using include center mode

        # we need to pad this size of buffer (max level of retina bufffer)
        self.pading_size = \
            (2 * self.radius * (self.retina_strides[0] ** n_retina),
             2 * self.radius * (self.retina_strides[1] ** n_retina))

    def get_dim(self, name):
        if name == 'glimpse':
            return (self.radius * 2) ** 2 * self.n_retina * self.channels
        else:
            raise ValueError

    @property
    def axis_order(self):
        # do not allow to change the order (no setter or deleter)
        return self.__axis_order

    def float2center_pixel(self, x, y):
        x_board, y_board = self.boarder
        # loc_x and loc_y is in the center ratangle
        loc_x = T.cast((self.img_width - self.radius * 2) *
                       (x - x_board[0]) // 2 + self.pading_size[0]//2,
                       'int32')
        loc_y = T.cast((self.img_height - self.radius * 2) *
                       (y - y_board[0]) // 2 + self.pading_size[1]//2,
                       'int32')
        return loc_x, loc_y

    def matrix2tensor4(self, img_2d):
        # only for input usuage
        batch_size = img_2d.shape[0]
        return img_2d.reshape((batch_size, self.channels, self.img_height,
                               self.img_width))

    def tensor4d2matrix(self, img_4d):
        # only for input usuage
        batch_size = img_4d.shape[0]
        return img_4d.reshape((batch_size,
                               self.channels * self.img_width *
                               self.img_height))

    def padding_img(self, img):
        batch_size = img.shape[0]
        img_paded = T.zeros((batch_size, self.channels,
                             self.img_height + self.pading_size[0],
                             self.img_width + self.pading_size[1]))
        offset = (self.pading_size[0] // 2, self.pading_size[1] // 2)
        # this expected a non-include center patch
        img_paded = T.set_subtensor(
            img_paded[:, :, offset[0]:self.img_height + offset[0],
                      offset[1]:self.img_width + offset[1]], img)
        return img_paded

    def read(self, img, center_x, center_y):
        # this acturately the upper left corner axis
        if img.ndim == 2:
            img = self.matrix2tensor4(img)

        batch_size = img.shape[0]
        loc_x, loc_y = self.float2center_pixel(center_x, center_y)
        img_paded = self.padding_img(img)
        retina, _ = theano.map(self.do_glimpes, sequences=[img_paded, loc_x,
                                                           loc_y])
        return retina.reshape((batch_size, self.get_dim('glimpse')))

    def do_glimpes(self, img, loc_x, loc_y):
        """
        a subpatch of an image
        """
        patch = img[:, loc_y - self.radius:loc_y + self.radius,
                    loc_x - self.radius:loc_x + self.radius]
        glimpe = self.retina(img, loc_x, loc_y)
        retina = T.concatenate([patch, glimpe], axis=0)
        return retina

    def filter(self, img, ds):
        return T.signal.pool.pool_2d(img, ds=ds)

    def do_retina(self, i, img, loc_x, loc_y):
        stride_x, stride_y = (self.radius * self.retina_strides[0] ** i,
                              self.radius * self.retina_strides[1] ** i)
        patch = img[:, loc_y - stride_y:loc_y + stride_y,
                    loc_x - stride_x:loc_x + stride_x]
        return self.filter(patch, (self.retina_strides[0] ** i,
                                   self.retina_strides[1] ** i))

    def retina(self, img, loc_x, loc_y):
        # we do not use scan for doing retina this is just for sytax sugar
        # retina, _ = theano.scan(self.do_retina,
        #                         sequences=[np.arange(1, self.n_retina)],
        #                         non_sequences=[img, loc_x, loc_y])
        retina = []
        for i in range(1, self.n_retina):
            retina.append(self.do_retina(i, img, loc_x, loc_y))
        retina = T.concatenate(retina, axis=0)
        return retina

    def nn2att(self, l):
        center_x = l[:, 0]
        center_y = l[:, 1]

        # center_x = T.tanh(center_x_emb)
        # center_y = T.tanh(center_y_emb)
        return center_x, center_y

    def att_mark(self, img, center_x, center_y):
        """
        This method is for monitoring the training behavior.
        """
        # action = action + 1  # making color is different from others (0, 1)
        if img.ndim == 2:
            img = self.matrix2tensor4(img)

        batch_size = img.shape[0]
        loc_x, loc_y = self.float2center_pixel(center_x, center_y)
        img_paded = self.padding_img(img)
        # convert to RGB
        img_paded = T.repeat(img_paded, 3, axis=1)

        retagle_idx_x, retagle_idx_y = img_ratangle(
            (self.radius * 2, self.radius * 2))

        def mark(img, loc_x, loc_y):
            for i in range(self.n_retina):
                x, _ = self.retina_strides
                rng = self.radius * x ** i
                retagle_idx_x, retagle_idx_y = \
                    img_ratangle(((rng) * 2,
                                  (rng) * 2))
                patch = img[:, loc_y - rng:loc_y + rng,
                            loc_x - rng:loc_x + rng]
                patch = T.set_subtensor(patch[0, retagle_idx_x, retagle_idx_y],
                                        255)
                patch = T.set_subtensor(patch[1:, retagle_idx_x, retagle_idx_y],
                                        0)

                img = T.set_subtensor(img[:, loc_y - rng:loc_y +
                                      rng, loc_x - rng:loc_x + rng], patch)
            return img

        mark_img, _ = theano.map(mark, sequences=[img_paded, loc_x,
                                                  loc_y])
        out = mark_img[:, :, self.pading_size[0]/2:-self.pading_size[0]/2,
                       self.pading_size[1]/2:-self.pading_size[1]/2].reshape((
                           batch_size, 3, self.img_height, self.img_width))
        return out


class DiscreteRetinaGlimpse(RetinaGlimpse):
    def __init__(self, **kwargs):
        super(DiscreteRetinaGlimpse, self).__init__(**kwargs)
        self.emb_dim = self.img_width * self.img_height

    def read(self, img, center_x, center_y):
        loc_x, loc_y = center_x, center_y
        if img.ndim == 2:
            img = self.matrix2tensor4(img)
        batch_size = img.shape[0]
        img_paded = self.padding_img(img)
        retina, _ = theano.map(self.do_glimpes, sequences=[img_paded, loc_x,
                                                           loc_y])
        return retina.reshape((batch_size, self.get_dim('glimpse')))


class GlimpseNetwork(Initializable):
    """
    GlimpseSensor & Linear + Rectifier
    ----------------------------------

    apply:
        input_shape (batch_size, n_channels * img_width * img_height)
        output_dim (batch_size, dim)

    """
    def __init__(self, dim,
                 n_channels, img_height, img_width, N, sensor=None,
                 n_retina=3, radius=4,
                 activations=None, **kwargs):

        super(GlimpseNetwork, self).__init__(**kwargs)
        if sensor is None or sensor == 'simple':
            self.sensor = GlimpseSensorBeta(channels=n_channels,
                                            img_height=img_height,
                                            img_width=img_width, N=N)
        elif sensor == 'retina':
            self.sensor = RetinaGlimpse(img_width, img_height, n_channels,
                                        n_retina=n_retina, radius=radius)
        else:
            raise ValueError("sensor mode support [simple]|[retina]." +
                             'Got ' + sensor + '.')

        self.loc_emb = self.sensor.emb_dim

        self.glimpes_0 = Linear(input_dim=self.loc_emb,
                                output_dim=dim,
                                name=self.name + '_glimp_0',
                                weights_init=self.weights_init,
                                biases_init=self.biases_init)

        self.glimpes_1 = Linear(input_dim=self.sensor.get_dim('glimpse'),
                                output_dim=dim, name=self.name + '_glimp_1',
                                weights_init=self.weights_init,
                                biases_init=self.biases_init)

        self.glimpes_out = Linear(input_dim=dim*2, output_dim=dim,
                                  name=self.name + '_glimp_out',
                                  weights_init=self.weights_init,
                                  biases_init=self.biases_init)

        self.children = [self.glimpes_0, self.glimpes_1, self.glimpes_out]
        self.output_dim = dim

    def get_dim(self, name):
        if name == 'img':
            return self.sensor.get_dim('img')
        elif name == 'l_last':
            return self.sensor.emb_dim
        else:
            raise ValueError

    @application(contexts=['img'], sequences=[],
                 state=['l_last'], outputs=['hidden_g'])
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
        l_unpack = self.sensor.nn2att(l_last)
        glimpes = self.sensor.read(img, *l_unpack)
        h0 = T.nnet.relu(self.glimpes_0.apply(l_last))
        h1 = T.nnet.relu(self.glimpes_1.apply(glimpes))
        h_c = T.concatenate([h0, h1], axis=1)
        hidden_g = T.nnet.relu(self.glimpes_out.apply(h_c))
        return hidden_g


class LocationNetwork(Random, Initializable):
    def __init__(self, input_dim, loc_emb, std, non_hetro=False, **kwargs):
        super(LocationNetwork, self).__init__(**kwargs)
        self.std = std
        self.non_hetro = non_hetro
        self.mean_affine = Linear(
                input_dim=input_dim,
                output_dim=loc_emb,
                weights_init=self.weights_init,
                biases_init=self.biases_init)

        if non_hetro:
            self.std_affine = Linear(input_dim=input_dim,
                                     output_dim=loc_emb,
                                     weights_init=self.weights_init,
                                     biases_init=self.biases_init)

            self.children = [self.mean_affine, self.std_affine]
        else:
            self.children = [self.mean_affine]

    def get_dim(self, name):
        if name == 'hidden_g':
            return self.transform.get_dim('inputs')
        elif name in ['l', 'l_sample']:
            return self.transform.get_dim('outputs')
        else:
            raise ValueError

    @application(inputs=['hidden_g'], outputs=['l', 'l_sample'])
    def apply(self, hidden_g):
        loc_mean = self.mean_affine.apply(hidden_g)
        loc_u = hard_tanh(loc_mean)
        if self.non_hetro:
            loc_std = T.nnet.relu(self.std_affine.apply(hidden_g))
            std = loc_std
        else:
            std = self.std
        loc_sample = self.theano_rng.normal(avg=loc_u,
                                            std=std,
                                            size=loc_mean.shape,
                                            dtype=theano.config.floatX)

        return loc_u, hard_tanh(loc_sample)


class CoreNetwork(BaseRecurrent, Initializable):
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

    @recurrent(sequences=['inputs'], states=['state', 'cell'], contexts=[],
               outputs=['state', 'cell'])
    def apply(self, inputs, state, cell):
        state, cell = self.lstm.apply(self.proj.apply(inputs), state, cell,
                                      iterate=False)
        return state, cell


class ActionNetwork(Initializable):
    def __init__(self, input_dim, n_classes, multi_object=False, **kwargs):
        super(ActionNetwork, self).__init__(**kwargs)
        self.transform = Linear(input_dim=input_dim,
                                output_dim=n_classes, **kwargs)
        if multi_object:
            self.out = Logistic()
        else:
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


class RAM(BaseRecurrent, Random, Initializable):
    """
    Recurrent Attention Model (RAM)

    Paramerters
    -----------
    core : core type layer
    step_output : which space to output
    """
    def __init__(self, core, glimpse_network, location_network,
                 action_network, n_steps, task_env='statics',
                 random_init_loc=True, **kwargs):

        super(RAM, self).__init__(**kwargs)
        self.core = core  # projec to hidden state
        self.glimpse_network = glimpse_network  # sensor information
        self.action_network = action_network  # action network
        self.location_network = location_network
        self.n_steps = n_steps + 1
        self.random_init_loc = random_init_loc
        self.task_env = task_env

        self.children = [self.glimpse_network, self.core,
                         self.location_network,
                         self.action_network]

    def random_init(self, batch_size, name):
        state_init = self.theano_rng.uniform(low=-1., high=1.,
                                             size=(batch_size,
                                                   self.get_dim(name)))
        return state_init

    def get_dim(self, name):
        if name in 'img':
            return self.glimpse_network.get_dim('img')
        elif name == 'l_last':
            return self.glimpse_network.loc_emb
        elif name == 'action':
            return self.action_network.output_dim
        elif name == 'state':
            return self.core.get_dim('state')
        elif name == 'cell':
            return self.core.get_dim('cell')
        else:
            raise ValueError

    @recurrent(sequences=[],
               contexts=['img'], states=['l_last', 'state', 'cell'],
               outputs=['l_last', 'action', 'state', 'cell', 'l_sample'])
    def apply(self, img, l_last, state, cell):
        hidden_g = self.glimpse_network.apply(img, l_last)
        state, cell = self.core.apply(hidden_g, state, cell, iterate=False)
        action = self.action_network.apply(state)
        l, l_sample = self.location_network.apply(state)
        return l_sample, action, state, cell, l

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *arg, **kwargs):
        results = []
        for state in self.apply.states:
            dim = self.get_dim(state)
            if dim == 0:
                raise ValueError('Currently do not upport dim = 0. Got dim 0'
                                 ' on variable %s'.format(state))
            if self.random_init_loc and state == 'l_last':
                res = self.random_init(batch_size, state)
            else:
                res = T.zeros((batch_size, dim))
            results.append(res)
        return results

    @application(inputs=['img'],
                 outputs=['l', 'action', 'state', 'cell', 'l_sample'])
    def out(self, img):
        batch_size = img.shape[0]
        l, action, state, cell, l_sample = self.apply(
            img, n_steps=self.n_steps,  # the last step is for prediction step
            batch_size=batch_size)
        return l, action, state, cell, l_sample

    @recurrent(sequences=[],
               contexts=['img'], states=['l_last', 'state', 'cell'],
               outputs=['l_last', 'action', 'state', 'cell', 'img_loc'])
    def detail_sample(self, img, l_last, state, cell):
        hidden_g = self.glimpse_network.apply(img, l_last)
        state, cell = self.core.apply(hidden_g, state, cell, iterate=False)
        action = self.action_network.apply(state)
        l = self.location_network.apply(state)
        sensor = self.glimpse_network.sensor

        loc = sensor.nn2att(l)
        W = sensor.read(img, *loc)
        img_loc = sensor.write(W, *loc)
        return l, action, state, cell, img_loc

    @application(inputs=['img'], outputs=['img_loc'])
    def sample(self, img):
        batch_size = img.shape[0]
        _, _, _, _, img_loc = self.detail_sample(img, n_steps=self.n_steps,
                                                 batch_size=batch_size)
        return img_loc


if __name__ == '__main__':
    from blocks.graph import ComputationGraph

    test = (np.random.normal(size=(100, 28 * 28 * 1)).astype(floatX),
            np.random.normal(size=(100, 2)).astype(floatX),
            np.random.normal(size=(100, 30)).astype(floatX),
            np.random.normal(size=(100, 30)).astype(floatX))

    inits = {
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }

    glim_net = GlimpseNetwork(dim=12,
                              n_channels=1, img_height=28,
                              img_width=28,
                              N=4, name='glimpes_net',
                              **inits)
    core = CoreNetwork(input_dim=12, dim=30,
                       name='core_net', **inits)
    loc_net = LocationNetwork(30, loc_emb=2, std=0.11,
                              name='loc_net', **inits)
    # Test loc net
    action = ActionNetwork(30, 4, **inits)
    ram = RAM(core, glim_net, loc_net, action, 30, name='RAM', **inits)
    ram.initialize()

    img = T.matrix('img')
    l = T.matrix('l_last')
    state = T.matrix('state')
    cell = T.matrix('cell')

    # Tests
    h_g = glim_net.apply(img, l)
    rval = h_g.eval({img: test[0], l: test[1]})
    assert rval.shape == (100, 12)
    print str(glim_net) + ' Pass ...'
    os = core.apply(h_g, state, cell, iterate=False)
    rvals = [o.eval({img: test[0], l: test[1], state:test[2], cell:test[3]})
             for o in os]
    print rvals

    print [rvali.shape for rvali in rvals]
    print str(core) + ' Pass ...'
    # rval = action.applys()
    fn = ram.out(img)

    cg = ComputationGraph(fn)
    out = theano.function([img], fn,
                          allow_input_downcast=True, updates=cg.updates)
    res = out(test[0])
    print [i for i in res][1]
