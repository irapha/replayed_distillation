import os
import numpy as np

from os import listdir
from tensorflow.python.platform import gfile
from skimage.transform import resize
from skimage.io import imread
from utils import grouper


class CASIAFingerprintIterator(object):

    def __init__(self):
        self.og = read_data_set("CASIA-FingerprintV5/")

    @property
    def io_shape(self):
        # read_data_set crops and rescales each image to 224x224
        return 224*224, 500

    def train_epoch_in_batches(self, batch_size):
        train_list = list(range(len(self.og['train']['images'])))
        np.random.shuffle(train_list)
        for batch_i in grouper(train_list, batch_size):
            batch = [(self.og['train']['images'][i],
                      self.og['train']['labels'][i])
                    for i in batch_i if i is not None]
            yield zip(*batch)

    def test_epoch_in_batches(self, batch_size):
        test_list = list(range(len(self.og['test']['images'])))
        np.random.shuffle(test_list)
        for batch_i in grouper(test_list, batch_size):
            batch = [(self.og['test']['images'][i],
                      self.og['test']['labels'][i])
                    for i in batch_i if i is not None]
            yield zip(*batch)

def read_data_set(image_dir):
    """Loads the casia dataset, crops and rescales to 224x224, subtracts the
    saved pixel means, shuffles and separates into train/test sets."""
    if not gfile.Exists(image_dir):
        raise Exception("Image directory '" + image_dir + "' not found.")
    base_classes = listdir(image_dir)
    class_count = len(base_classes)
    result = {'train': {'images': [], 'labels': []},
              'test': {'images': [], 'labels': []}}
    pixel_means = np.load('datasets/casia_pixel_means.npy')

    for i, label in enumerate(base_classes):
        print('reading images for class {}/{}'.format(i + 1, class_count), end='\r')
        base_left = os.path.join(image_dir, label, 'L')
        base_right = os.path.join(image_dir, label, 'R')
        examples = ([os.path.join(base_left, x) for x in listdir(base_left) if x[-3:] == 'bmp'] +
                [os.path.join(base_right, y) for y in listdir(base_right) if y[-3:] == 'bmp'])
        np.random.shuffle(examples)

        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[int(label)] = 1.0

        breakpoint = int(len(examples) * 0.4)

        for example in examples[breakpoint:]:
            result['train']['images'].append(crop_rescale(imread(example)) - pixel_means)
            result['train']['labels'].append(ground_truth)


        for example in examples[:breakpoint]:
            result['test']['images'].append(crop_rescale(imread(example)) - pixel_means)
            result['test']['labels'].append(ground_truth)
    print('') # for the new line
    return result

def crop_rescale(image):
    # original image is 356x328
    # first crop to 328x328
    image = image[14:342,0:328]
    # then rescale to 224x224
    image = resize(image, (224, 224), mode='constant')
    # finally, flatten it
    return np.reshape(image, (224*224,))

