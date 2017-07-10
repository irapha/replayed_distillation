import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
from utils import grouper


class OptimizedDatasetIterator(object):

    def __init__(self, dataset_location):
        self.data_optimized = np.load(dataset_location)[()]

    @property
    def io_shape(self):
        # data_optimized[0] is all batches for class = 0
        # data_optimized[0][0] is the first batch for class = 0, as a tuple
        # data_optimized[0][0][0] is the first item in that tuple (batch_x)
        # data_optimized[0][0][0][0] is the first example in that batch
        # data_optimized[0][0][1] is the second item in that tuple (batch_y) which is a list of len 1
        # data_optimized[0][0][1][0] is the single element in that list, which is the actual batch of batch_size
        # data_optimized[0][0][1][0][0] is the first latent in that batch of latents
        return len(self.data_optimized[0][0][0][0]), len(self.data_optimized[0][0][1][0][0])

    def train_epoch_in_batches(self, _):
        classes_and_batches = [(clas_idx, batch_idx)
                for clas_idx in range(self.io_shape[1])
                for batch_idx  in range(len(self.data_optimized[0]))]
        np.random.shuffle(classes_and_batches)

        for clas_idx, batch_idx in classes_and_batches:
            yield self.data_optimized[clas_idx][batch_idx]
