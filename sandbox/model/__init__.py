import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from pylearn2.models.mlp import Layer
from util.topology import Node


##########################################################
# How to use pylearn2.models.Model                       #
# Overider the method which you want to implement first  #
##########################################################


class LayerLab(Layer):
    """ Abstract Class for Layer
        Meta Class Only Define the basic topology operation
    """
    _parent_node = None
    _params = {}
    _inbound_node = None
    # Add other extension like: dropout _inbound add a Drop Layer
    _input_dim = None
    _output_dim = None

    def set_input_dim(self, input_dim):
        self._input_dim = input_dim

    def set_output_dim(self, output_dim):
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
                layer = self
            else:
                if configs:
                    assert n_layers == len(configs)
                    assert self.get_output_space.dim == configs[-1]
                    input_dim = layer.input_space
                    output_dim = configs[ith]
                else:
                    input_dim = layer.get_output_space()
                    output_dim = layer.get_output_space()
                layer_last = layer
                layer_next = self.set_input_space(layer_last.get_output_space())
                layer_next.add(layer_last)
                layer = layer_next

        self._inbound_node = layer
        return self
