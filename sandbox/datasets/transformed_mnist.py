""" Transform the mnist dataset into a bigger image, to test the effect of
attention.
"""
from fuel.datasets.hdf5 import H5PYDataset
import h5py
import numpy as np
import datasets

image_size, channels, data_train, data_valid, data_test = \
    datasets.get_data('mnist')

data_sources = np.vstack([data_train.data_sources[0],
                          data_test.data_sources[0],
                          data_valid.data_sources[0]])

data_targets = np.vstack([data_train.data_sources[1],
                          data_test.data_sources[1],
                          data_valid.data_sources[1]])

batch_size = data_sources.shape[0]
data_buffer = np.zeros((batch_size, 1, 60, 60))

# create the index
start = np.random.random_integers(
    low=0, high=60 - 28, size=(batch_size, 2)) + 28 // 2
img_idx = np.arange(60 * 60).reshape(60, 60)


for i, j in zip(range(batch_size), start):
    x, y = j.tolist()
    data_buffer[i, :, x-14: x+14, y-14: y+14] = \
        data_sources[i, :, :, :]

data_buffer = data_buffer.astype('float32')
data_targets = data_targets.astype('int32')

f = h5py.File('mnist_transform.hdf5', mode='w')
features = f.create_dataset('features', (80000, 1, 60, 60), dtype='float32')
targets = f.create_dataset('targets', (80000, 1), dtype='int32')
# assign the data
features[...] = data_buffer
targets[...] = data_targets

split_dict = {
    'train': {'features': (0, 60000),
              'targets': (0, 60000)},
    'valid': {'features': (60000, 70000),
              'targets': (60000, 70000)},
    'test': {'features': (70000, 80000),
             'targets': (70000, 80000)}
              }
f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
f.flush()
f.close()
