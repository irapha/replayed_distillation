
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
    else:
        raise NotImplemented('This dataset not implemented')

def get_io_size(dataset_name, procedure):
    if dataset_name in ['mnist', 'mnist_bottleneck']:
        if procedure in ['train_conv', 'replay_conv']:
            return 1024, 10
        return 784, 10
    else:
        raise NotImplemented('This dataset not implemented')
