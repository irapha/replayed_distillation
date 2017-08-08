import numpy as np

from skimage.transform import resize
from utils import grouper


class CASIAPalmRegionIterator(object):

    def __init__(self):
        #  self.og = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.og = read_data_set("CASIA-PalmprintV1/")

    @property
    def io_shape(self):
        # TODO
        return None, None

    def train_epoch_in_batches(self, batch_size):
        train_list = list(range(len(self.og.train.images)))
        np.random.shuffle(train_list)
        for batch_i in grouper(train_list, batch_size):
            batch = [(np.reshape(resize(
                             np.reshape(self.og.train.images[i], (28, 28)),
                             (32, 32), mode='constant'), (1024,)),
                      self.og.train.labels[i])
                    for i in batch_i if i is not None]
            yield zip(*batch)

    def test_epoch_in_batches(self, batch_size):
        test_list = list(range(len(self.og.test.images)))
        np.random.shuffle(test_list)
        for batch_i in grouper(test_list, batch_size):
            batch = [(np.reshape(resize(
                             np.reshape(self.og.test.images[i], (28, 28)),
                             (32, 32), mode='constant'), (1024,)),
                      self.og.test.labels[i])
                    for i in batch_i if i is not None]
            yield zip(*batch)

def read_data_set(dataset_dir):
    # dataset dir should be filled w/ dirs, each representing a person, and
    # each having images of that person's palms
    raise NotImplemented("Rapha, finish this.")
