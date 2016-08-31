from pylearn2.space import SimplyTypedSpace, CompositeSpace, VectorSpace
#                            Conv2DSpace)
from pylearn2.utils import wraps
import numpy as np
import theano
import theano.tensor as T
from theano.gof.op import get_debug_values
from pylearn2.sandbox.rnn.space import (SequenceSpace, SequenceDataSpace)


class CompositeSpaceLab(CompositeSpace):
    def __init__(self, **kwargs):
        super(CompositeSpaceLab, self).__init__(**kwargs)
        self.components = self._flatten_components(self.components)

    def _flatten_components(self, components):
        components_stack = []
        for space in components:
            if isinstance(space, CompositeSpace):
                assert isinstance(space.components, (tuple, list))
                components_stack.extend(space.components)
            else:
                components_stack.append(space)
        return components_stack


class SequenceSpaceLab(SequenceSpace):
    def make_theano_batch(self):
        return super(CompositeSpace, self).make_batch_theano()

class SequenceDataSpaceLab(SimplyTypedSpace):
    """
    Make a space (time_steps, Space)
    """


class ContextSpace(SimplyTypedSpace):
    def __init__(self, dim, num_annotation,
                 dtype='floatX', **kwargs):
        super(ContextSpace, self).__init__(dtype, **kwargs)
        self.dim = dim
        self.num_annotation = num_annotation

    def __str__(self):
        return '%s(dim=%i, num_annotation=%i, dtype=%s)' % (
            self.__class__.__name__, self.dim, self.num_annotation,
            str(self.dtype))

    def __eq__(self, other):
        if not isinstance(other, ContextSpace):
            return False
        return (self.dim, self.num_annotation) == (other.dim,
                                                   other.num_annotation)

    @wraps(SimplyTypedSpace._validate_impl)
    def _validate_impl(self, is_numeric, batch):
        super(ContextSpace, self)._validate_impl(is_numeric, batch)
        if is_numeric:
            if batch.ndim != 3:
                raise TypeError("ContectSpace should have a 3D array. Got " +
                                str(batch.ndim))
        else:
            if not isinstance(batch, theano.gof.Variable):
                raise TypeError("Not a valid syblic variable. Got " +
                                str(batch))
            if batch.ndim != 3:
                raise TypeError("Required a 3D tensor. Got " + str(batch) +
                                " with %i" % batch.ndim)
            for val in get_debug_values(batch):
                self.np_validate(val)

    @wraps(SimplyTypedSpace.get_origin_batch)
    def get_origin_batch(self, batch_size, dtype=None):
        dtype = self._clean_dtype_arg(dtype)
        return np.zeros(shape=(batch_size, self.num_annotation,
                               self.dim)).astype(dtype)

    @wraps(SimplyTypedSpace._format_as_impl)
    def _format_as_impl(self, is_numeric, batch, space):
        if space == self:
            return batch
        else:
            if isinstance(space, SequenceDataSpace):
                if is_numeric:
                    formatted_batch = np.transpose(np.asarray([
                        self.space._format_as_impl(is_numeric, sample,
                                                   space.space)
                        for sample in np.transpose(batch, (1, 0, 2))
                    ]), (1, 0, 2))
                else:
                    formatted_batch, _ = theano.scan(
                        fn=lambda elem: self.space._format_as_impl(
                            is_numeric, elem, space.space),
                        sequences=[batch]
                    )
                return formatted_batch
            elif isinstance(space, VectorSpace):
                # Convert to Vector Space do not required reshape
                result = batch
                return space._cast(result, space.dtype)
            else:
                print('Unexpected space', space)
                raise NotImplementedError

    @wraps(SimplyTypedSpace.get_total_dimension)
    def get_total_dimension(self):
        return self.dim * self.num_annotation

    @wraps(SimplyTypedSpace.make_theano_batch)
    def make_theano_batch(self, name=None, dtype=None, batch_size=None):
        return T.tensor3(name=name, dtype=dtype)

# XXX Add support for Converting Conv2D to Context Space,
