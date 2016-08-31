import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import numpy as np

from pylearn2.space import VectorSpace, Conv2DSpace, CompositeSpace
from pylearn2.utils import wraps, sharedX
from model import LayerLab
from space import ContextSpace
# from model.mlp_hook import AttentionWrapper
# from pylearn2.models.mlp import MLP, Softmax, SpaceConverter


class Attention(LayerLab):
    """
    Attention Parent Class
    ======================
    Attention is a MLP to generate alpha dis from given event happend. Alpha
    distribution is called the Attention distribution which indictes the
    attention is located.

    Input Space
    -----------
    3D tensor (batch, n_annotations, context dim) (ContextSpace)
    2D tensor (batch, dim) (VectorSpace)

    Output Space
    ------------
    A 2D tensor (batch, n_annotations)

    NOTE:
        Default is SoftAttention
    """
    def __init__(self, layers,
                 layer_name, irange=None, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.layers = layers
        self.lstm = None  # the owner of an attention object
        self.layer_names = set()
        for layer in self.layers:
            if layer.layer_name in self.layer_names:
                raise ValueError("attention layers found dublicated layer"
                                 " names")
            self.layer_names.add(layer.layer_name)
        self.irange = irange
        # setting rng (use the host rng)

    @wraps(LayerLab.set_input_space)
    def set_input_space(self, space):

        self.input_space = space
        self.required_reformat = False
        if isinstance(space, CompositeSpace):
            if tuple(type(i) for i in space.components) != \
                    (ContextSpace, VectorSpace):
                if tuple(type(i) for i in space.components) == \
                        (Conv2DSpace, VectorSpace):
                    self.required_reformat = True
                    num_annotation = space.shape[0] * space.shape[1]
                    dim = space.num_channels
                    self.desired_space = CompositeSpace([
                        ContextSpace(dim=dim, num_annotation=num_annotation),
                        space.components[1]])
                else:
                    raise TypeError("Currently do not support Non-ContextSpace"
                                    "or Convolition input space. "
                                    "CompositeSpace([ContextSpace, "
                                    "VectorSpace]). Got " + str(space))

        else:
            raise TypeError("Attention need "
                            "CompositeSpace([ContextSpace, VectorSpace])")

        self.output_space = self.input_space.components[0]
        # setup for nested attention
        self._update_layer_input_spaces()
        context_space, state_space = self.input_space.components

        rng = self.lstm.mlp.rng
        # initial parameters
        state_dim = state_space.dim
        context_dim = context_space.dim

        self.Wd_att = sharedX(rng.uniform(-self.irange, self.irange,
                                          (state_dim, context_dim)),
                              name=self.layer_name + '_Wd_att')
        self.Ud_att = sharedX(rng.uniform(-self.irange, self.irange,
                                          (context_dim, 1)),
                              name=self.layer_name + '_Ud_att')
        self.bd_att = sharedX(np.zeros((1,)),
                              name=self.layer_name + '_bd_att')
        self._params.append(self.Wd_att, self.Ud_att, self.bd_att)

    def set_lstm(self, lstm):
        assert self.lstm is None
        self.lstm = lstm

    def _update_layer_input_spaces(self):
        """
        Check a valid alpha space, and Update the space of project layers
        ---

        """
        layers = self.layers

        context, state = self.input_space.components

        layers[0].set_input_space(context)
        for i in range(1, len(layers)):
            layers[i].set_input_space(layers[i-1].get_output_space())

        if not layers[-1].get_output_space().dim == context.dim:
            raise RuntimeError("The Last layer of projection Layer doesn't "
                               "match the context dim. Context dim is %i. "
                               "Got %i" % (context.dim,
                                           layers[-1].get_output_space().dim))

    def _project(self, context, return_all=False):
        self.input_space.components[0].validate(context)
        rval = context
        rval_list = []
        for i in range(len(self.layers)):
            rval = self.layers[i].fprop(rval)
            rval_list.append(rval)

        if return_all:
            return rval_list
        return rval

    def pre_alpha(self, state, pctx):
        """
        state, projected_context
        """
        self.input_space.validate((pctx, state))
        pstate = T.dot(state, self.Wd_att)
        # project the state into context dimention
        pctx = pctx + pstate[:, None, :]
        pctx = T.tanh(pctx)
        pre_alpha = T.dot(pctx, self.Ud_att) + self.bd_att
        pre_alpha_shape = pre_alpha.shape
        return pre_alpha.reshape((pre_alpha_shape[0], pre_alpha[1]))

    def alpha(self, state, pctx):
        """
        Compute the Alpha Distribution
        Do something about
        ex:
            pre_alpha = self.pre_alpha(state, pcontext)
            return T.nnet.sigmoid(pre_alpha)
        """
        raise NotImplementedError

    def alpha_sample(self, state, pctx, context):
        """
        Attention Sample:
        """
        raise NotImplementedError


class SoftAttention(Attention):

    @wraps(Attention.alpha)
    def alpha(self, state, pctx):
        self.input_space.validate((pctx, state))
        pre_alpha = self.pre_alpha(state, pctx)
        return T.nnet.softmax(pre_alpha)

    @wraps(Attention.alpha_sample)
    def alpha_sample(self, state, pctx, context):
        self.input_space.validate((pctx, state))
        alpha = self.alpha(state, pctx)
        return (context * alpha[:, :, None]).sum(1)


class HardAttention(Attention):

    def __init__(self, inverse_temperture=1., semi_sampling_p=0.5,
                 use_sampling=True, use_argmax=False, **kwargs):
        super(HardAttention, self).__init__(**kwargs)
        assert not all(use_sampling, use_argmax)

        self.inverse_temperture = sharedX(np.float32(inverse_temperture),
                                          name=self.layer_name +
                                          '_temperture_c')

        self.semi_sampling_p = sharedX(np.float32(semi_sampling_p),
                                       name=self.layer_name +
                                       '_semi_sampling_p')
        try:
            self.theano_rng = MRG_RandomStreams(123)  # GPU rng
        except:
            self.theano_rng = T.shared_randomstreams.RandomStreams(123)

        self.use_sampling = use_sampling
        self.use_argmax = use_argmax

    def alpha(self, state, pctx):
        pre_alpha = self.inverse_temperture * self.pre_alpha(state, pctx)
        return T.nnet.sigmoid(pre_alpha)

    def argmax_alpha_sample(self, state, pctx, context):
        alpha = self.alpha(state, pctx)
        alpha_max = T.max(self.alpha, axis=1, keepdim=True)
        return T.cast(T.eq(T.arrange(alpha.shape[1])[None, :], alpha_max),
                      'float32')

    def random_alpha_sample(self, state, pctx, context):
        alpha = self.alpha(state, pctx)
        alpha_sample = self.theano_rng.multinominal(pvals=alpha,
                                                    dtype='float32')
        semi_mask = self.theano_rng.binomianl(n=1,
                                              p=self.semi_sampling_p,
                                              size=(1,)).sum()

        return semi_mask * alpha_sample + (1 - semi_mask) * alpha_sample

    def alpha_sample(self, state, pctx, context):
        self.input_space.components[0].validate(context)
        self.input_space.validate((pctx, state))
        if self.use_sampling:
            return self.random_alpha_sample(state, pctx, context)

        elif self.use_argmax:
            return self.argmax_alpha_sample(state, pctx, context)


