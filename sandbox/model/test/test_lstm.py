import theano
import theano.tensor as T
import numpy as np
from PIL import Image

from pylearn2.space import CompositeSpace
from pylearn2.models.mlp import MLP
from pylearn2.space import IndexSpace
from pylearn2.sandbox.rnn.space import SequenceSpace, SequenceMaskSpace
from pylearn2.models.mlp import Linear
from pylearn2.models.mlp import CompositeLayer
from pylearn2.utils import safe_zip

from space import ContextSpace
from util.tests import Im2LatexTest
from model.lstm import ConditionLSTM
from model.highway import HighWay
# from pylearn2.sandbox.rnn.models.rnn import LSTM
from model.lstm import ContextLSTM, GlimpseSensor
from model.embedings import WordEmbeding
from model.im2latex import Seq2Seq, MeanReduce, DataPass


flatten = lambda lst: reduce(lambda l, i: l + flatten(i)
         if isinstance(i, (list, tuple)) else l + [i], lst, [])


#@Im2LatexTest.test_layers
def test_Seq2Seq_set_input_space(dataset):
    # input_space =
    # Seq2Seq(decoder_layers=, bridge_layers=, encoder_layers=,)
    def debug(self, name=None, dtype=None, batch_size=None):
        if name is None:
            name = [None] * len(self.components)
        elif not isinstance(name, (list, tuple)):
            name = ['%s[%i]' % (name, i) for i in xrange(len(self.components))]

        dtype = self._clean_dtype_arg(dtype)

        assert isinstance(name, (list, tuple))
        assert isinstance(dtype, (list, tuple))

        rval = tuple([x.make_theano_batch(name=n,
                                          dtype=d,
                                          batch_size=batch_size)
                      for x, n, d in safe_zip(self.components,
                                              name,
                                              dtype)])
        return rval

    SequenceSpace.make_theano_batch = debug

    input_space = CompositeSpace(
        [
            ContextSpace(dim=512, num_annotation=14*14),
            CompositeSpace([IndexSpace(dim=123, max_labels=10),
                            SequenceMaskSpace()])
         ]
    )

    input = input_space.make_theano_batch()

    encoder_layers = [
        ContextLSTM(dim=31, irange=0.2, layer_name='encoder_lstm')
    ]

    bridge_layers = [
        MeanReduce(axis=1, layer_name='mean_reduce_layer'),
        HighWay(layer_name='fflayer'),
        Linear(dim=10 * 2, irange=0.2, layer_name='proj_state_memory')
    ]

    embedings = WordEmbeding(dim=100, layer_name='embeding')

    decoder_layers = [
        ConditionLSTM(dim=10, irange=0.2, layer_name='cond_lstm')
    ]

    seq2seq = Seq2Seq(encoder_layers=encoder_layers,
                      bridge_layers=bridge_layers,
                      decoder_layers=decoder_layers,
                      embedings=embedings,
                      input_space=input_space,
                      maxlen_encoder=14*14, maxlen_decoder=123,
                      annotation_dim=512, max_labels=10,
                      layer_name='seq2seq')
    #print seq2seq.decoder_layers[0].context_space.dim
    print seq2seq
    #np_input = flatten(input_space.get_origin_batch(12))
    #np_input[-1] = np.zeros((123, 12)).astype('float32')
    #input = map(lambda x:theano.shared(x), np_input)
    out = seq2seq.fprop(input)
    input = flatten(input)
    np_input = flatten(input_space.get_origin_batch(12))
    np_input[-1] = np.zeros((123, 12)).astype('float32')  #TODO: Create a new space...
    fn = theano.function(input, out)
    print fn(*np_input)[0].shape
    print seq2seq.layer_names


def test_Seq2Seq_set_input_space(dataset):
    # input_space =
    # Seq2Seq(decoder_layers=, bridge_layers=, encoder_layers=,)
    def debug(self, name=None, dtype=None, batch_size=None):
        if name is None:
            name = [None] * len(self.components)
        elif not isinstance(name, (list, tuple)):
            name = ['%s[%i]' % (name, i) for i in xrange(len(self.components))]

        dtype = self._clean_dtype_arg(dtype)

        assert isinstance(name, (list, tuple))
        assert isinstance(dtype, (list, tuple))

        rval = tuple([x.make_theano_batch(name=n,
                                          dtype=d,
                                          batch_size=batch_size)
                      for x, n, d in safe_zip(self.components,
                                              name,
                                              dtype)])
        return rval

    SequenceSpace.make_theano_batch = debug

    input_space = CompositeSpace(
        [
            ContextSpace(dim=512, num_annotation=14*14),
            IndexSpace(dim=123, max_labels=10),
            SequenceMaskSpace()
         ]
    )

    input = input_space.make_theano_batch()

    encoder_layers = [
        CompositeLayer(layer_name='composite_encoder',
                       layers=[
            ContextLSTM(dim=31, irange=0.2, layer_name='encoder_lstm'),
            DataPass(layer_name='data_pass')
        ],
                       inputs_to_layers={0: [0], 1: [1], 2:[1]})
    ]

    bridge_layers = [
        MeanReduce(axis=1, layer_name='mean_reduce_layer'),
        HighWay(layer_name='fflayer'),
        Linear(dim=10 * 2, irange=0.2, layer_name='proj_state_memory')
    ]

    bridge_wrapper = MLP(layer_name='bridge',
                         layers=bridge_layers)

    CompositeLayer('composite_bridge', layers=[
        DataPass(layer_name='data_pass'), bridge_wrapper],
        inputs_to_layers={0: [0], 1: [1], 2:[1]})

    embedings = CompositeLayer(
        layers=[WordEmbeding(dim=100, layer_name='embeding'),
                DataPass('data_pass')], inputs_to_layers={0: 1, 1:1, 2:0})

    decoder_layers = [
        ConditionLSTM(dim=10, irange=0.2, layer_name='cond_lstm')
    ]

    seq2seq = Seq2Seq(encoder_layers=encoder_layers,
                      bridge_layers=bridge_layers,
                      decoder_layers=decoder_layers,
                      embedings=embedings,
                      input_space=input_space,
                      maxlen_encoder=14*14, maxlen_decoder=123,
                      annotation_dim=512, max_labels=10,
                      layer_name='seq2seq')
    #print seq2seq.decoder_layers[0].context_space.dim
    print seq2seq
    #np_input = flatten(input_space.get_origin_batch(12))
    #np_input[-1] = np.zeros((123, 12)).astype('float32')
    #input = map(lambda x:theano.shared(x), np_input)
    out = seq2seq.fprop(input)
    input = flatten(input)
    np_input = flatten(input_space.get_origin_batch(12))
    np_input[-1] = np.zeros((123, 12)).astype('float32')  #TODO: Create a new space...
    fn = theano.function(input, out)
    print fn(*np_input)[0].shape
    print seq2seq.layer_names


@Im2LatexTest.call_test
def test_glimpse():

    N = 40
    channels = 3
    height = 480
    width = 640

    # ------------------------------------------------------------------------
    att = GlimpseSensor(channels, height, width, size=N)

    I_ = T.matrix()
    center_y_ = T.vector()
    center_x_ = T.vector()
    W_ = att.apply(I_, center_y_, center_x_)

    do_read = theano.function(inputs=[I_, center_y_, center_x_],
                              outputs=W_, allow_input_downcast=True)
    print 'Build Done'
    # W_ = T.matrix()
    # center_y_ = T.vector()
    # center_x_ = T.vector()
    # delta_ = T.vector()
    # sigma_ = T.vector()
    # I_ = att.write(W_, center_y_, center_x_, delta_, sigma_)
    # do_write = theano.function(inputs=[W_, center_y_, center_x_, delta_,
    # sigma_],
    #                          outputs=I_, allow_input_downcast=True)

    # ------------------------------------------------------------------------

    I = Image.open("../../../../stat-learning/Stat_Comp/draw/draw/cat.jpg")
    I = I.resize((640, 480))  # .convert('L')

    I = np.asarray(I).transpose([2, 0, 1])
    I = I.reshape((channels*width*height))
    I = I / 255.

    center_y = 100.5
    center_x = 330.5
    delta = 5.
    sigma = 2.

    def vectorize(*args):
        return [a.reshape((1,)+a.shape) for a in args]

    I, center_y, center_x, delta, sigma = \
        vectorize(I, np.array(center_y),
                  np.array(center_x), np.array(delta), np.array(sigma))

    # import ipdb; ipdb.set_trace()

    W = do_read(I, center_y, center_x, delta, sigma)
    #I2 = do_write(W, center_y, center_x, delta, sigma)

    def imagify(flat_image, h, w):
        image = flat_image.reshape([channels, h, w])
        image = image.transpose([1, 2, 0])
        return image / image.max()

    import pylab
    pylab.figure()
    pylab.gray()
    pylab.imshow(imagify(I, height, width), interpolation='nearest')

    pylab.figure()
    pylab.gray()
    pylab.imshow(imagify(W, N, N), interpolation='nearest')

    pylab.figure()
    pylab.gray()
    pylab.imshow(imagify(I2, height, width), interpolation='nearest')
    pylab.show(block=True)

