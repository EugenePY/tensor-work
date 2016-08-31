from pylearn2.space import Space, VectorSpace
from util.tests import Im2LatexTest
import theano
import theano.tensor as T
from theano.gof.op import get_debug_values
import numpy as np


@Im2LatexTest.call_test
def test_space_std():
    theano.config.compute_test_value = 'warn'
    x = T.imatrix('x')
    x.tag.test_value = np.zeros((12, 12))
    print get_debug_values(x)
