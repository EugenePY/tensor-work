"""
Simple computation graph helper:
    Dealing high level deep learing Layers behaviors.
"""
from collections import OrderedDict


def merge(layer, layer1):
    """
    Merge two Node into one.
    """
    pass


class GraphContainer(object):
    """
    Compose Layers
    """
    _container = OrderedDict()

    def __str__(self):
        return self._container

    @property
    def container(self):
        return self._container

    def stack(self, layer, prefix=None):
        if prefix:
            prefix = str(layer)
        if len(self._container) == 0:
            self._container[prefix] = layer
        else:
            connected_layer = layer.add(self._container.popitem()[1])
            self._container[prefix] = connected_layer
        return self


class Node(object):
    _input_dim = (None)
    _output_dim = (None)
    _inbound_node = None

    def get_parent(self):
        """
        return the last Nodes
        """
        raise NotImplementedError

    def __add__(self, b):
        """
        Connect two node
        """
        return self.connect(b)

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    def connect(self, b):
        raise NotImplementedError


class Layer(Node):
    """
    Perform Matrix Operation:
    Soft Connection:
    """
    _parent_node = None
    _params = {}
    _inbound_node = None
    # Add other extension like: dropout _inbound add a Drop Layer

    def __init__(self, input_dim, output_dim):
        self._input_dim = input_dim
        self._output_dim = output_dim

    def __str__(self):
        return "{0}.{1}".format(self.__class__.__name__, str(self.__hash__()))

    def __init_params__(self):
        """
        Set the Parameters of a Layer
        """
        raise NotImplementedError

    @property
    def parent_node(self):
        return self._parent_node

    @parent_node.setter
    def parent_node(self, node):
        self._check_valid_parent_node(node)
        self._parent_node = node

    def add(self, node):
        self.parent_node = node

    @parent_node.deleter
    def parent_node(self):
        self._parent_node = None

    def get_graph(self):
        node_list = [self]
        while node_list[-1].get_parent() is not None:
            node_list.append(node_list[-1].get_parent())
        return node_list[::-1]

    def get_parent(self):
        return self._parent_node

    def connect(self, b):
        self.parent_node = b

    def _check_valid_parent_node(self, node):
        assert isinstance(node, Node)
        assert len(list([node])) == 1
        assert node.output_dim == self._input_dim

    def stacking(self, n_layers, configs=None):
        for ith in range(n_layers):
            if ith == 0:
                input_dim = self._input_dim
                output_dim = self._output_dim
                layer = Layer(input_dim, output_dim)
            else:
                if configs:
                    assert n_layers == len(configs)
                    assert self._output_dim == configs[-1]
                    input_dim = layer.output_dim
                    output_dim = configs[ith]
                else:
                    input_dim = layer.output_dim
                    output_dim = layer.output_dim
                layer_last = layer
                layer_next = Layer(input_dim, output_dim)
                layer_next.add(layer_last)
                layer = layer_next
        self._inbound_node = layer
        return self

    @property
    def inbound_node(self):
        return self._inbound_node


class InputNode(Node):
    _parent_node = None


class Layer3D(Layer):
    """
    Perform Convolution Operation
    """
    pass


class HybridLayer(Layer):
    pass


class DeepLayer(Layer):
    def __init__(self, input_dim, output_dim, n_layers, configs=None):
        self._input_dim = input_dim
        self._output_dim = output_dim

        for ith in range(n_layers):
            if ith == 0:
                input_dim = input_dim
                output_dim = output_dim
                layer = Layer(input_dim, output_dim)
            else:
                if configs:
                    assert n_layers == len(configs)
                    assert self._output_dim == configs[-1]
                    input_dim = layer.output_dim
                    output_dim = configs[ith]
                else:
                    input_dim = layer.output_dim
                    output_dim = layer.output_dim
                layer_last = layer
                layer_next = Layer(input_dim, output_dim)
                layer_next.add(layer_last)
                layer = layer_next

        self._inbound_node = layer

    @property
    def inbound_node(self):
        return self._inbound_node

    def output(self):
        raise NotImplementedError
