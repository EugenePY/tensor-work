from pylearn2.models.mlp import RectifiedLinear, CompositeLayer, MLP

from pylearn2.sandbox.rnn.models.rnn import LSTM
from pylearn2.sandbox.rnn.space import SequenceDataSpace
from pylearn2.space import VectorSpace
from pylearn2.sandbox.rnn.models.rnn import RNN

input_space = SequenceDataSpace(VectorSpace(dim=20))

# simple attention
# the location network take a glimpes of the images
location_network = MLP(layer_name='location_network', layers=[
    RectifiedLinear(layer_name='rectifier', dim=2)])

glimps_sensor = MLP(layer_name='glimps_sensor', layers=[
    RectifiedLinear(layer_name='rectifier', dim=3)])


glimps_network = CompositeLayer(layer_name='glimps_network',
                                layers=[location_network, glimps_sensor],
                                inputs_to_layers={0:[0], 1:[1]})

rnn = RNN(layers=[LSTM(layer_name='lstm0', dim=12,
                       irange=0.2)], input_space=input_space)


# train with REINFORCE algorithm
