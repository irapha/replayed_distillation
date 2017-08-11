import numpy as np

from os import listdir
from tensorflow.python.platform import gfile
from skimage.transform import resize
from utils import grouper


class CASIAPalmRegionIterator(object):

    def __init__(self):
        #  self.og = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.og = read_data_set("CASIA-FingerprintV5/")

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

def create_image_lists(image_dir):
    """Builds a list of training images from the file system.
    Analyzes the sub folders in the image directory, splits them into stable
    training, testing, and validation sets, and returns a data structure
    describing the lists of images for each label and their paths.
    Args:
        image_dir: String path to a folder containing subfolders of images.
    Returns:
        A dictionary containing an entry for each label subfolder, with images split
        into training, testing, and validation sets within each label.
    """
    if not gfile.Exists(image_dir):
        raise Exception("Image directory '" + image_dir + "' not found.")
    result = {}
    base_classes = listdir(image_dir)
    print('base cls: {}'.format(base_classes))
    raise Exception()

    for label in base_classes:
        examples = listdir(os.path.join(image_dir, label))
        np.random.shuffle(examples)
        example_list = []

        for example in examples:
            if example[:3] != 'ASL':
                print('found bad path: ' +
                        os.path.join(image_dir, label, example))

            image_list = sorted(map(lambda x: '/'.join(x.split('/')[-2:]),
                    glob.glob(os.path.join(image_dir, label, example, '*'))))

            example_list.append(image_list)

        breakpoint = int(len(example_list) / 2)
        result[label] = {'dir': label,
                # this ensures that all overflow goes to training.
                'training': example_list[breakpoint:],
                'testing': example_list[:breakpoint]}
    return result

create_image_lists('CASIA-FingerprintV5')
