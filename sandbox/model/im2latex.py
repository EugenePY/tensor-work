# Pylearn2
from pylearn2.utils import wraps
from pylearn2.space import CompositeSpace, VectorSpace, IndexSpace
from pylearn2.sandbox.rnn.space import (SequenceSpace, SequenceDataSpace,
                                        SequenceMaskSpace)
from pylearn2.sandbox.rnn.models.rnn import RNN

# Numpy
import numpy as np
import theano
import theano.tensor as T

# Local libs
#from model.highway import HighWay, FeedWay
#from model.output import SoftMax
#from model.lstm import LSTMwithAttention, LSTM
from model import LayerLab
from space import ContextSpace
from cost import Seq2SeqCost
from util.topology import GraphContainer


class Seq2Seq(RNN, LayerLab):
    """
    input dim: (max input len, batch_size, input_dim)
    output dim: (max output len, batch_size, output_dim)
    """
    _topo = GraphContainer()
    seed = 123
    def __init__(self, encoder_layers, bridge_layers,
                 decoder_layers, input_space,
                 maxlen_encoder, maxlen_decoder, annotation_dim, max_labels,
                 layer_name, embedings=None,
                 batch_size=None,
                 input_source='features', target_source='targets',
                 **kwargs):
        assert len(encoder_layers) >= 1
        assert len(decoder_layers) >= 1
        assert len(bridge_layers) >= 1
        assert maxlen_encoder is not None
        assert maxlen_decoder is not None
        assert annotation_dim is not None

        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.bridge_layers = bridge_layers
        self._input_source = input_source
        self._trarget_sourece = target_source

        # Seq2Seq map Seq1 feature  2 Seq2 target
        self.layer_name = layer_name
        self._nested = False
        self.layer_names = set()
        self.batch_size = batch_size
        self.force_batch_size = batch_size
        self._input_source = input_source
        self._target_source = target_source

        if embedings:
            use_embeding = True
            self.embedings = embedings
            self.embedings.set_mlp(self)
            self.layer_names.add(self.embedings.layer_name)
        else:
            use_embeding = False

        self.use_embeding = use_embeding
        self.layers = self.encoder_layers + self.bridge_layers + \
            self.decoder_layers

        #self.monitor_targets = monitor_targets
        for layer in self.layers:
            assert layer.get_mlp() is None
            if layer.layer_name in self.layer_names:
                raise ValueError("Seq2Seq.__init__ given two or more layers "
                                 "with same name: " + layer.layer_name)

            layer.set_mlp(self)

            self.layer_names.add(layer.layer_name)

        if input_space is not None:  # XXX this part is not general. need to
                                # dealing some normal case
            self.setup_rng()
            if self.use_embeding:
                self.input_space = CompositeSpace(
                    [ContextSpace(dim=annotation_dim,
                                num_annotation=maxlen_encoder),
                     CompositeSpace([IndexSpace(max_labels=max_labels,
                                            dim=maxlen_decoder),
                                     SequenceMaskSpace()])])
            else:
                self.input_space = CompositeSpace(
                    [ContextSpace(dim=annotation_dim,
                                num_annotation=maxlen_encoder),
                     SequenceSpace(IndexSpace(max_labels=max_labels,
                                            dim=maxlen_decoder))])
            assert self.input_space == input_space
            self.context_space, self.seq_space = self.input_space.components
            self._update_layer_input_spaces()

    def _update_layer_input_spaces(self):
        layers = self.encoder_layers + self.bridge_layers
        self.encoder_layers[0].set_input_space(self.context_space)

        for i in range(1, len(layers)):
            layers[i].set_input_space(layers[i-1].get_output_space())

        if self.use_embeding:
            self.embedings.set_input_space(self.seq_space.components[0])
            seq_space = SequenceSpace(self.embedings.get_output_space().space)
        else:
            seq_space = self.seq_space

        self.decoder_layers[0].set_input_space(
            CompositeSpace([seq_space, self.context_space]))

        for i in range(1, len(self.decoder_layers)):
            self.decoder_layers[i].set_input_space(
                self.decoder_layers[i-1].get_output_space())

    @wraps(RNN.fprop)
    def fprop(self, state_below):
        self.input_space.validate(state_below)
        context, (x, mask) = state_below
        encode = context
        for encoder in self.encoder_layers + self.bridge_layers:
            encode = encoder.fprop(encode)

        if self.use_embeding:
            x = self.embedings.upward_pass(x)

        cond = (x, mask, context)
        for decoder in self.decoder_layers:
            cond = decoder.fprop(cond, z0=encode)

        return cond

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


class MeanReduce(LayerLab):
    def __init__(self, axis, layer_name, output_space=VectorSpace, **kwargs):
        self.axis = axis
        self.layer_name = layer_name
        self._output_space = output_space

    def set_input_space(self, space):
        assert isinstance(space, ContextSpace)
        self.input_space = space
        self.output_space = self._output_space(dim=space.dim)

    def fprop(self, state_below):
        out = T.mean(state_below, axis=self.axis)
        self.output_space.validate(out)
        return out


class DataPass(LayerLab):
    def __init__(self, layer_name):
        self.layer_name = layer_name

    def set_input_space(self, space):
        self.input_space = space
        self.output_space = self.input_space

    def fprop(self, state_below):
        self.input_space.validate(state_below)
        return state_below


class BadInputSpaceError(TypeError):
    """
    Handle the bad input space
    """

if __name__ == "__main__":
    pass

