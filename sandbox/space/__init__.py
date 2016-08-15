from pylearn2.space import SimplyTypedSpace, VectorSpace
from pylearn2.utils import wraps
import numpy as np
import theano
import theano.tensor as T


class ContextSpace(VectorSpace):
    def __init__(self, dim, num_annotation,
                 dtype='floatX', **kwargs):
        super(ContextSpace, self).__init__(dtype, **kwargs)
        self.dim = dim
        self.num_annotation = num_annotation

    @wraps(VectorSpace.get_origin_batch)
    def get_origin(self, batch_size):
        return np.zeros(shape=(batch_size, self.num_annotation,
                               self.dim))

    @wraps(VectorSpace.format_as)
    def format_as(self, batch, space):
        if not isinstance(space, VectorSpace):
            raise TypeError("ContextSpace do not support format_as " +
                            str(space))
        assert self.dim == 1
        return batch.reshape((batch.shape[0], batch.shape[1]))

    @wraps(VectorSpace.get_total_dimension)
    def get_total_dimension(self):
        return self.dim

    @wraps(VectorSpace.make_theano_batch)
    def make_theano_batch(self, name=None, dtype=None, batch_size=None):
        return T.tensor3(name=name, dtype=dtype)

    @wraps(VectorSpace._validate_impl)
    def _validate_impl(self, is_numeric, batch):
        super(VectorSpace, self)._validate_impl(is_numeric, batch)
        if isinstance(batch, theano.gof.Variable):
            if batch.ndim != 3:
                raise ValueError('ContextSpace only support 3D tensor at ' +
                                 str(batch))
        else:
            raise ValueError(str(batch) + " is not a tensor variable")
