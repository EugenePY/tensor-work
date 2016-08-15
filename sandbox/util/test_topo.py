from util.tests import test
from util.topology import Node, Layer, DeepLayer, DeepTopology


@test
def test_Layer():
    layer = Layer(input_dim=100, output_dim=10)
    layer_next = Layer(input_dim=10, output_dim=2)
    layer_next.add(layer)
    assert layer_next.get_parent() == layer
    assert layer_next.get_parent().input_dim == 100
    assert layer_next.get_parent().output_dim == 10
    assert layer_next.input_dim == 10
    assert layer_next.output_dim == 2

@test
def test_deepcompose():
    deep = DeepLayer(input_dim=100, output_dim=10, n_layers=13)
    assert len(deep.inbound_node.get_graph()) == 13
    assert deep.input_dim == 100
    assert deep.output_dim == 10

def test_topo():
    DeepGraph()

