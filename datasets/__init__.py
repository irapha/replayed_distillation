
def get(dataset_name, bottleneck_file=None):
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
    if dataset_name == 'mnist_bottleneck':
        from . import mnist_bottleneck as m
        return m.MNISTIterator(bottleneck_file)
    else:
        raise NotImplemented('This dataset not implemented')

def get_io_size(dataset_name):
    if dataset_name in ['mnist', 'mnist_bottleneck']:
        return 784, 10
    else:
        raise NotImplemented('This dataset not implemented')
