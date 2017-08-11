
def get(dataset_name):
    """ Returns a dataset object with easy interface.

    Interface:
        #  dataset.train.next_batch(batch_size)
        #  dataset.test.next_batch(batch_size)

        #  dataset.train.images # all images
        #  dataset.train.labels # all labels
    """
    if dataset_name == 'mnist':
        from . import mnist as m
        return m.MNISTIterator()
    if dataset_name == 'mnist_conv':
        from . import mnist_conv as m
        return m.MNISTResizedIterator()
    if dataset_name == 'casia':
        from . import casia as c
        return c.CASIAFingerprintIterator()
    else:
        from . import optimized_dataset as d
        return d.OptimizedDatasetIterator(dataset_name)
