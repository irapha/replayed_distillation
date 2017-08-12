import os
import numpy as np

from os import listdir
from tensorflow.python.platform import gfile
from skimage.transform import resize
from skimage.io import imread
from utils import grouper


class YaleFacesIterator(object):

    def __init__(self):
        self.og = read_data_set("ExtendedYaleB/")
        self.pixel_means = np.load('datasets/yale_pixel_means.npy')

    @property
    def io_shape(self):
        # read_preprocess crops and rescales each image to 224x224
        return 224*224, 28

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
        """Reads image path from image_lists, crops, rescales to 224x224,
        and subtracts the saved pixel means"""
        return crop_rescale(imread(img)) - self.pixel_means

def read_data_set(image_dir):
    """Loads the yale dataset as image lists, shuffles and separates into
    train/test sets."""
    if not gfile.Exists(image_dir):
        raise Exception("Image directory '" + image_dir + "' not found.")
    base_classes = listdir(image_dir)
    class_count = len(base_classes)
    result = {'train': {'images': [], 'labels': []},
              'test': {'images': [], 'labels': []}}

    total_images = 0
    for i, label in enumerate(base_classes):
        print('reading images for class {}/{}'.format(i + 1, class_count), end='\r')
        base_dir = os.path.join(image_dir, label)
        examples = [os.path.join(base_dir, x) for x in listdir(base_dir) if x[-3:] == 'pgm']
        total_images += len(examples)
        np.random.shuffle(examples)

        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[i] = 1.0

        breakpoint = int(len(examples) * 0.4)

        for example in examples[breakpoint:]:
            result['train']['images'].append(example)
            result['train']['labels'].append(ground_truth)

        for example in examples[:breakpoint]:
            result['test']['images'].append(example)
            result['test']['labels'].append(ground_truth)
    print('') # for the new line
    print('total images: {}'.format(total_images))
    return result

def crop_rescale(image):
    # original image is 640x480
    # first crop to 480x480
    image = image[80:560,0:480]
    # then rescale to 224x224
    image = resize(image, (224, 224), mode='constant')
    # finally, flatten it
    return np.reshape(image, (224*224,))

def save_pixel_means(image_dir):
    """Loads the yale dataset as images, crops, rescales, shuffles and
    separates into train/test sets."""
    if not gfile.Exists(image_dir):
        raise Exception("Image directory '" + image_dir + "' not found.")
    base_classes = listdir(image_dir)
    class_count = len(base_classes)
    result = []

    for i, label in enumerate(base_classes):
        print('reading images for class {}/{}'.format(i + 1, class_count), end='\r')
        base_dir = os.path.join(image_dir, label)
        examples = [os.path.join(base_dir, x) for x in listdir(base_dir) if x[-3:] == 'pgm']
        np.random.shuffle(examples)

        breakpoint = int(len(examples) * 0.4)

        for example in examples[breakpoint:]:
            result.append(crop_rescale(imread(example)))

    print('') # for the new line

    mean = np.mean(result, axis=0)
    np.save('datasets/yale_pixel_means.npy', mean)
    print('shape: {}'.format(np.shape(mean)))
    return result

if __name__ == '__main__':
    save_pixel_means('ExtendedYaleB/')
