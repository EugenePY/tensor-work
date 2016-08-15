from dataset.im2latex import Im2LatexData
# from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from util.tests import Im2LatexTest


@Im2LatexTest.call_test
def test_dataset():
    dataset = Im2LatexData(which_set='debug')
    iteration_mode = 'shuffled_sequential'
    for batch in dataset.iterator(mode=iteration_mode, batch_size=2):
        assert all([ba.shape == (2, 10) for ba in batch[:-1]] +
                   [batch[-1].shape==(2, 13, 14, 15)])

