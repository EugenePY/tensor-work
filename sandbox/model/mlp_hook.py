"""
Letting RNN and MLP Layers support ContextSpace input
"""
from pylearn2.utils.track_version import MetaLibVersion
from pylearn2.utils import wraps
from pylearn2.space import VectorSpace
from pylearn2.sandbox.rnn.space import SequenceDataSpace, SequenceSpace
from space import ContextSpace



class AttentionWrapper(MetaLibVersion):
    def __new__(cls, name, bases, dct):
        wrappers = [attr[:-8] for attr in cls.__dict__.keys()
                    if attr.endswith('_wrapper')]
        for wrapper in wrappers:
            if wrapper not in dct:
                for base in bases:
                    method = getattr(base, wrapper, None)
                    if method is not None:
                        break
            else:
                method = dct[wrapper]
            dct[wrapper] = getattr(cls, wrapper + '_wrapper')(name, method)

        dct['seq2seq_friendly'] = False
        dct['_requires_reshape'] = False
        dct['_requires_unmask'] = False
        dct['_input_space_before_reshape'] = None
        return type.__new__(cls, name, bases, dct)

    @classmethod
    def set_input_space_wrapper(cls, name, set_input_space):
        @wraps(set_input_space)
        def outer(self, input_space):
            if not self.seq2seq_friendly:
                if isinstance(input_space, ContextSpace):
                    self._requires_reshape = True
                    self._input_space_before_reshape = input_space
                    input_space = SequenceDataSpace(
                        VectorSpace(dim=input_space.dim))
                    self.output_space = SequenceDataSpace(
                        VectorSpace(dim=self.dim))
                if isinstance(input_space, (SequenceSpace, SequenceDataSpace)):
                    pass
                else:
                    raise TypeError("Current Seq2Seq LSTM do not support "
                                    "none-context space. Got " +
                                    str(input_space))
            return set_input_space(self, input_space)
        return outer

    @classmethod
    def fprop_wrapper(cls, name, fprop):
        @wraps(fprop)
        def outer(self, state_below, return_all=False):
            if self._requires_reshape:
                if isinstance(state_below, tuple):
                    ndim = state_below[0].ndim
                    reshape_size = state_below[0].shape
                else:
                    ndim = state_below.ndim
                    reshape_size = state_below.shape
                inp_shape = (reshape_size[1], reshape_size[0], reshape_size[2])
                output = fprop(self, state_below.reshape(inp_shape),
                             return_all=return_all)
                output_shape = output.shape
                output = output.reshape((output_shape[1], output_shape[0],
                                       output_shape[2]))
                self.output_space.validate(output)
                return output
            else:
                return fprop(self, state_below, return_all=return_all)
        return outer

    @classmethod
    def get_output_space_wrapper(cls, name, get_output_space):
        """
        Same thing as set_input_space_wrapper.

        Parameters
        ----------
        get_output_space : method
            The get_output_space method to be wrapped
        """
        @wraps(get_output_space)
        def outer(self):
            if (not self.seq2seq_friendly and self._requires_reshape and
                not isinstance(get_output_space(self), ContextSpace)):
                if isinstance(self._input_space_before_reshape, ContextSpace):
                    return ContextSpace(dim=get_output_space(self).dim,
                                   num_annotation=\
                        self._input_space_before_reshape.num_annotation)
            else:
                return get_output_space(self)
        return outer


if __name__ == "__main__":
    # simple test it seems ok....
    from pylearn2.sandbox.rnn.models.rnn import LSTM
    class LSTM_CONTEXT(LSTM): __metaclass__ = AttentionWrapper
    print LSTM_CONTEXT.fprop
