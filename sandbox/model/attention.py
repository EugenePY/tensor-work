from pylearn2.models.mlp import MLP, Softmax, SpaceConverter
from pylearn2.space import VectorSpace
from pylearn2.sandbox.rnn.models.mlp_hook import RNNWrapper
from pylearn2.utils import wraps
import theano.tensor as T

from space import ContextSpace
from model.highway import LinearAttention
# from model.mlp_hook import AttentionWrapper


class Attention(MLP):
    """
    Attention Parent Class
    ======================
    Attention is a MLP to generate alpha dis from given event happend. Alpha
    distribution is called the Attention distribution which indictes the
    attention is located.

    Input Space
    -----------
    A 3D tensor (batch, n_annotations, dim) (ContextSpace)

    Output Space
    ------------
    A 2D tensor (batch, n_annotations)

    NOTE:
        Default is SoftAttention
    """
    def __init__(self, layers, **kwargs):
        assert isinstance(kwargs['input_space'], ContextSpace)
        super(Attention, self).__init__(layers=layers, **kwargs)

        self.output_layers = \
            [LinearAttention(dim=1, layer_name='reduct_dim_to_1_layer',
                             irange=1),
             SpaceConverter(layer_name='attention_space_convert',
                           output_space=VectorSpace(
                               dim=self.input_space.num_annotation)),
             Softmax(n_classes=self.get_input_space().num_annotation,
                    layer_name='attention_output', irange=.2)]

        self.add_layers(self.output_layers)
        self.project_layer = self.layers[:-4]
        self.alpha_layer = self.layers[-4:]

    @wraps(MLP.set_input_space)
    def set_input_space(self, space):
        if not isinstance(space, ContextSpace):
            raise TypeError("Currently do not support Non-ContextSpace"
                            "input space.")
        if hasattr(self, "mlp"):
            assert self._nested
            self.rng = self.mlp.rng
            self.batch_size = self.mlp.batch_size

        self.input_space = space
        # setup for nested attention
        self._update_layer_input_spaces()

    def project(self, state_below):
        self.input_space.validate(state_below)
        pctx = self.fprop(state_below, return_all=True)[-4]
        return pctx

    def alpha_dis(self, pctx):
        self.alpha_layer[0].get_input_space().validate(pctx)
        state_below = pctx
        for layer in self.output_layers:
            state_below = layer.fprop(state_below)
        alpha = state_below
        return alpha

class SoftAttention(Attention): pass


class HardAttention(Attention):
    def __init__(self, dim, mcmc, input_space=None, selector=None,
                 temperature=.87,
                 attn_type='stochastic'):
        pass

##### Declare Attention support Layers
