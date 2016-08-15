# Pylearn2
from pylearn2.utils import wraps
from pylearn2.space import CompositeSpace, VectorSpace
from pylearn2.sandbox.rnn.space import (SequenceSpace, SequenceDataSpace)
from pylearn2.sandbox.rnn.models.rnn import RNN

# Numpy
import numpy as np

# Local libs
#from model.highway import HighWay, FeedWay
#from model.output import SoftMax
#from model.lstm import LSTMwithAttention, LSTM
from model import LayerLab
from cost import Seq2SeqCost
from util.topology import GraphContainer


class Seq2Seq(RNN, LayerLab):
    """
    input dim: (max input len, batch_size, input_dim)
    output dim: (max output len, batch_size, output_dim)
    """
    _topo = GraphContainer()

    def __init__(self, encoder_layers, bridge_layers,
                 decoder_layers, input_space=None,
                 batch_size=None, maxlen_encoder=None, maxlen_decoder=None,
                 input_source='features', target_source='targets',
                 layer_name=None, **kwargs):
        self.layers = encoder_layers + decoder_layers + bridge_layers
        assert len(encoder_layers) >= 1
        assert len(decoder_layers) >= 1
        assert len(bridge_layers) >= 1
        assert maxlen_encoder is not None
        assert maxlen_decoder is not None

        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self._input_source = input_source
        self._trarget_sourece = target_source
        # Seq2Seq map Seq1 feature  2 Seq2 target

        if input_space is None:  # XXX this part is not general need to
                                # dealing some normal case
            self.input_space = CompositeSpace(
                [SequenceDataSpace(space=VectorSpace(dim=maxlen_encoder)),
                 SequenceSpace(dim=maxlen_decoder)])

    @wraps(RNN.fprop)
    def fprop(self, state_below):
        encode = state_below
        for encoder in self.encoder_layers + self.bridge_layers:
            encode = encoder.fprop(encode)

        cond = encoder
        for decoder in self.decoder_layers:
            cond = decoder.fprop(cond)

        return cond

    def set_rng(self):
        return np.random.RandomState(self.seed)

    @wraps(RNN.get_input_space)
    def get_output_space(self):
        return self.layers[-1].get_target_space()

    @wraps(RNN.get_default_cost)
    def get_default_cost(self):
        return Seq2SeqCost()

    @wraps(RNN.get_monitoring_channels)
    def get_monitoring_channels(self, data):
        if self.monitor_targets:
            seq1, seq2 = data
        else:
            seq1 = data
            seq2 = None

        rvel = self.get_layer_moniror_channel(state_below=seq1, targets=seq2)
        return rvel


class BadInputSpaceError(TypeError):
    """
    Handle the bad input space
    """

if __name__ == "__main__":
    pass

