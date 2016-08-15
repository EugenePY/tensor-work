from pylearn2.datasets.dataset import Dataset
from pylearn2.utils import wraps
from pylearn2.utils.iteration import (
    FiniteDatasetIterator, resolve_iterator_class)
from pylearn2.space import CompositeSpace, VectorSpace, Conv2DSpace
from pylearn2.utils.rng import make_np_rng
from pylearn2.sandbox.rnn.space import SequenceDataSpace, SequenceSpace
# from space import MatrixSpace
import numpy
N = numpy  # just for naming

#VectorSequenceSpace.get_batch_axis = lambda x: 1

class Im2LatexData(Dataset):
    """
    DataSet for Im2latex
    --------------------
    caps :
    """
    _default_seed = (10, 123, 33)

    def __init__(self, which_set='debug', start=None, end=None, shuffle=True,
                 lazy_load=False, rng=_default_seed):

        assert which_set in ['debug', 'train', 'test']
        if which_set == 'debug':
            maxlen, n_samples, n_annotations, n_features = 10, 12, 13, 14
            X = N.zeros(shape=(n_samples, maxlen))
            X_mask = X  # same with X
            Z = N.zeros(shape=(n_annotations, n_samples, n_features))
        elif which_set == 'train':
            pass
        else:
            pass

        self.X, self.X_mask, self.Z = (X, X_mask, Z)
        self.sources = ('features', 'target')

        self.spaces = CompositeSpace([
            SequenceSpace(space=VectorSpace(dim=self.X.shape[1])),
            SequenceDataSpace(space=VectorSpace(dim=self.Z.shape[-1]))
        ])
        self.data_spces = (self.spaces, self.sources)
        # self.X_space, self.X_mask_space, self.Z_space
        # Default iterator
        self._iter_mode = resolve_iterator_class('sequential')
        self._iter_topo = False
        self._iter_target = False
        self._iter_data_specs = self.data_spces
        self.rng = make_np_rng(rng, which_method='random_intergers')

    def __prepare_dataset(self, caps, features, worddict, maxlen=None,
                          n_words=10000, zero_pad=False):
        seqs = []
        feat_list = []
        for cc in caps:
            seqs.append([worddict[w] if worddict[w] < n_words else 1
                         for w in cc[0].split()])
            feat_list.append(features[cc[1]])

        lengths = [len(s) for s in seqs]

        if maxlen is not None:
            new_seqs = []
            new_feat_list = []
            new_lengths = []
            for l, s, y in zip(lengths, seqs, feat_list):
                if l < maxlen:
                    new_seqs.append(s)
                    new_feat_list.append(y)
                    new_lengths.append(l)
            lengths = new_lengths
            feat_list = new_feat_list
            seqs = new_seqs

            if len(lengths) < 1:
                return None, None, None

        self.Y = numpy.zeros((len(feat_list),
                              feat_list[0].shape[1])).astype('float32')
        for idx, ff in enumerate(feat_list):
            self.Y[idx, :] = numpy.array(ff.todense())
        self.Y = self.Y.reshape([self.Y.shape[0], 14*14, 512])
        if zero_pad:
            self.Y_pad = numpy.zeros((y.shape[0],
                                      y.shape[1]+1,
                                      y.shape[2])).astype('float32')
            self.Y_pad[:, :-1, :] = self.Y
            self.Y = self.Y_pad

        self.n_samples = len(seqs)
        self.maxlen = numpy.max(lengths)+1

        self.X = numpy.zeros((self.maxlen, self.n_samples)).astype('int64')
        self.X_mask = numpy.zeros((self.maxlen,
                                   self.n_samples)).astype('float32')
        for idx, s in enumerate(seqs):
            self.X[:lengths[idx], idx] = s
            self.X_mask[:lengths[idx]+1, idx] = 1.

    @wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 rng=None, data_spces=None, return_tuple=False):
        (mode, batch_size, num_batches, rng, data_spces) = \
            self._init_iterator(mode, batch_size, num_batches, rng, data_spces)
        return FiniteDatasetIterator(self, mode(self.get_num_examples(),
                                                batch_size, num_batches,
                                                rng), data_spces)

    def get_data(self):
        return (self.X, self.X_mask, self.Z)

    @wraps(Dataset.get_num_examples)
    def get_num_examples(self):
        return self.X.shape[0]

    def get_data_specs(self):
        return self.data_spces
