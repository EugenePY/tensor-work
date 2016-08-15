from util.tests import Im2LatexTest
from model.highway import HighWay
from theano.tensor.basic import TensorVariable


@Im2LatexTest.test_layers
def test_highway(dataset):

    layer = HighWay(dim=10, layer_name='HighwayToHell')
    space = dataset.get_data_specs()[0].components[-1]
    layer.set_input_space(space)

    assert isinstance(layer.fprop(space.make_theano_batch()), TensorVariable)
