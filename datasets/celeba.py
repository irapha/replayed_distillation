import os
import numpy as np

from os import listdir
from tensorflow.python.platform import gfile
from skimage.transform import resize
from skimage.io import imread
from utils import grouper


class CelebAFacesIterator(object):

    def __init__(self):
        self.og = read_data_set("CelebA/")
        #  self.pixel_means = self.calculate_pixel_means()

    @property
    def io_shape(self):
        # read_preprocess crops and rescales each image to 224x224
        return 224*224*3, 40*2 # each of the 40 attrs get binary 1-hot label

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

    def calculate_pixel_means(self):
        weight = 1.0 / len(self.og['train']['images'])
        mean = crop_rescale(imread(self.og['train']['images'][0])) * weight
        for i, img in enumerate(self.og['train']['images']):
            if i % 100 == 0 or i > 162702:
                print('calculating pixel means. reading image: {}/162701'.format(i+1), end='\r')
            mean += crop_rescale(imread(img)) * weight
        print('') # for the new line
        return mean

    def read_preprocess(self, img):
        """Reads image path from image_lists, crops, rescales to 224x224,
        and subtracts the saved pixel means"""
        return crop_rescale(imread(img))# - self.pixel_means

def read_data_set(image_dir):
    """Loads the casia dataset as image lists, shuffles and separates into
    train/test sets."""
    if not gfile.Exists(image_dir):
        raise Exception("Image directory '" + image_dir + "' not found.")

    result = {'train': {'images': [], 'labels': []},
              'test': {'images': [], 'labels': []}}

    # create partitions based on partitions file.
    # we'll use them to put each image, label pair in the right dict within result
    partitions = {}
    with open(os.path.join(image_dir, 'list_eval_partition.txt'), 'r') as f:
        for i, line in enumerate(f):
            img_name, partition = line.strip().split(' ')
            if int(img_name[:6]) != i + 1:
                raise ValueError('Parse error.')
            partition = 0 if int(partition) == 0 else 1 # 0 is train, 1 is test
            partitions[img_name] = partition

    with open(os.path.join(image_dir, 'list_attr_celeba.txt'), 'r') as f:
        f.readline()
        attribute_names = f.readline().strip().split(' ')
        for i, line in enumerate(f):
            if i % 100 == 0 or i == 202598:
                print('reading attributes for image: {}/202599'.format(i+1), end='\r')
            fields = line.strip().replace('  ', ' ').split(' ')
            img_name = fields[0]
            if int(img_name[:6]) != i + 1:
                raise ValueError('Parse error.')
            # one hot per attr
            attrs = [[1, 0] if a == -1 else [0, 1] for a in map(int, fields[1:])]
            attr_vec = np.reshape(np.array(attrs), (80,))
            # original, each attr gets -1 or 1
            #  attr_vec = np.array(list(map(int, fields[1:])))

            partition = 'train' if partitions[img_name] == 0 else 'test'
            result[partition]['images'].append(os.path.join(image_dir, 'img_align_celeba', img_name))
            result[partition]['labels'].append(attr_vec)

    print('') # for the new line
    return result

def crop_rescale(image):
    # original image is 178x218
    # first crop to 178x178
    image = image[20:198,0:178] / 255.0
    # then rescale to 224x224
    image = resize(image, (224, 224), mode='constant')
    # finally, flatten it
    return np.reshape(image, (224*224*3))

