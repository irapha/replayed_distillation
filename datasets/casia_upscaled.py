import os
import numpy as np

from os import listdir
from tensorflow.python.platform import gfile
from skimage.transform import resize
from skimage.io import imread
from utils import grouper


class CASIAFingerprintIterator(object):

    def __init__(self):
        self.og = read_data_set("datasets/CASIA-FingerprintV5/")

    @property
    def io_shape(self):
        # read_preprocess crops and rescales each image to 448x448
        return 448*448, 500

    def train_epoch_in_batches(self, batch_size):
        train_list = list(range(len(self.og['train']['images'])))
        np.random.shuffle(train_list)
        for batch_i in grouper(train_list, batch_size):
            batch = [(self.read_preprocess(self.og['train']['images'][i]),
                      self.og['train']['labels'][i])
                    for i in batch_i if i is not None]
            yield zip(*batch)

    def test_epoch_in_batches(self, batch_size):
        test_list = list(range(len(self.og['test']['images'])))
        np.random.shuffle(test_list)
        for batch_i in grouper(test_list, batch_size):
            batch = [(self.read_preprocess(self.og['test']['images'][i]),
                      self.og['test']['labels'][i])
                    for i in batch_i if i is not None]
            yield zip(*batch)

    def read_preprocess(self, img):
        """Reads image path from image_lists, crops, rescales to 448x448,
        and subtracts the saved pixel means"""
        return crop_rescale(imread(img))# - self.pixel_means

def read_data_set(image_dir):
    """Loads the casia dataset as image lists, shuffles and separates into
    train/test sets."""
    if not gfile.Exists(image_dir):
        raise Exception("Image directory '" + image_dir + "' not found.")
    base_classes = listdir(image_dir)
    class_count = len(base_classes)
    result = {'train': {'images': [], 'labels': []},
              'test': {'images': [], 'labels': []}}

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
            result['train']['images'].append(example)
            result['train']['labels'].append(ground_truth)

        for example in examples[:breakpoint]:
            result['test']['images'].append(example)
            result['test']['labels'].append(ground_truth)
    print('') # for the new line
    return result

def crop_rescale(image):
    # original image is 356x328
    # first crop to 328x328
    image = image[14:342,0:328]
    # then rescale to 448x448
    image = resize(image, (448, 448), mode='constant')
    # finally, flatten it
    return np.reshape(image, (448*448,))

