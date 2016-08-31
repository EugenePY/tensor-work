import numpy
from pylearn2.utils import sharedX


def ortho_weight(ndim):
    """
    Random orthogonal weights

    Used by norm_weights(below), in which case, we
    are ensuring that the rows are orthogonal
    (i.e W = U \Sigma V, U has the same
    # of rows, V has the same # of cols)
    """
    W = numpy.random.randn(ndim, ndim)
    u, _, _ = numpy.linalg.svd(W)
    return sharedX(u.astype('float32'))


def norm_weight(shape, scale=0.01, ortho=True):
    """
    Random weights drawn from a Gaussian
    """
    assert len(shape) == 2
    nin, nout = shape
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = numpy.random.randn(nin, nin)
        W, _, _ = numpy.linalg.svd(W)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return sharedX(W.astype('float32'))


def zero_init(shape):
    return sharedX(numpy.zeros(shape=shape).astype('float32'))
