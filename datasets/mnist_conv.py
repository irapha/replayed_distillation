import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
from skimage.transform import resize
from utils import grouper


class MNISTResizedIterator(object):

    def __init__(self):
        self.og = input_data.read_data_sets("MNIST_data/", one_hot=True)

    @property
    def io_shape(self):
        return 1024, 10

    def train_epoch_in_batches(self, batch_size):
        train_list = list(range(len(self.og.train.images)))
        np.random.shuffle(train_list)
        for batch_i in grouper(train_list, batch_size):
	    batch = [(np.expand_dims(resize(
			     np.reshape(self.og.train.images[i], (28, 28)),
			     (size, size), mode='constant'), axis=3),
		      self.og.train.labels[i])
		    for i in batch_i if i is not None]
            yield zip(*batch)

    def test_epoch_in_batches(self, batch_size):
        test_list = list(range(len(self.og.test.images)))
        np.random.shuffle(test_list)
        for batch_i in grouper(test_list, batch_size):
	    batch = [(np.expand_dims(resize(
			     np.reshape(self.og.test.images[i], (28, 28)),
			     (size, size), mode='constant'), axis=3),
		      self.og.test.labels[i])
		    for i in batch_i if i is not None]
            yield zip(*batch)
