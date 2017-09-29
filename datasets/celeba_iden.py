import os
import numpy as np

from os import listdir
from tensorflow.python.platform import gfile
from skimage.transform import resize
from skimage.io import imread
from utils import grouper
from collections import Counter


NUM_IDENTITIES_TO_USE = 300

class CelebAFacesIterator(object):

    def __init__(self):
        self.og = read_data_set("CelebA/")
        #  self.pixel_means = self.calculate_pixel_means()

    @property
    def io_shape(self):
        # read_preprocess crops and rescales each image to 224x224
        return 224*224*3, NUM_IDENTITIES_TO_USE

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

    # NOTE: we cant rely on partitions file for identities identification because
    # each identity is either entirely on train or entirely on test set..........
    # create partitions based on partitions file.
    # we'll use them to put each image, label pair in the right dict within result
    #  partitions = {}
    #  with open(os.path.join(image_dir, 'list_eval_partition.txt'), 'r') as f:
        #  for i, line in enumerate(f):
            #  img_name, partition = line.strip().split(' ')
            #  if int(img_name[:6]) != i + 1:
                #  raise ValueError('Parse error.')
            #  partition = 0 if int(partition) == 0 else 1 # 0 is train, 1 is test
            #  partitions[img_name] = partition

    with open(os.path.join(image_dir, 'identity_CelebA.txt'), 'r') as f:
        identities = []
        for i, line in enumerate(f):
            fields = line.strip().replace('  ', ' ').split(' ')
            identities.append(fields[1])
        most_common_iden, counts = zip(*Counter(identities).most_common(NUM_IDENTITIES_TO_USE))
        print('num_images: {}'.format(sum(counts)))
        identities_to_idx = {iden: i for i, iden in enumerate(most_common_iden)}

    flipped = np.zeros([NUM_IDENTITIES_TO_USE])

    with open(os.path.join(image_dir, 'identity_CelebA.txt'), 'r') as f:
        for i, line in enumerate(f):
            if i % 100 == 0 or i == 202598:
                print('reading attributes for image: {}/202599'.format(i+1), end='\r')
            fields = line.strip().replace('  ', ' ').split(' ')
            assert len(fields) == 2
            img_name = fields[0]
            if fields[1] not in most_common_iden:
                continue
            attrs = np.zeros([NUM_IDENTITIES_TO_USE])
            attrs[identities_to_idx[fields[1]]] = 1.0
            attr_vec = np.reshape(np.array(attrs), (NUM_IDENTITIES_TO_USE,))

            # this guarantees 3 images per identity in test set, rest in train set.
            partition = 'train'
            # this is the number of examples to put in test set. since we have ~30 ex/identity, 9 is 30%
            if flipped[identities_to_idx[fields[1]]] < 9:
                partition = 'test'
                flipped[identities_to_idx[fields[1]]] += 1

            result[partition]['images'].append(os.path.join(image_dir, 'img_align_celeba', img_name))
            result[partition]['labels'].append(attr_vec)

    print('') # for the new line
    print('train_images: {} test_images: {}'.format(len(result['train']['images']), len(result['test']['images'])))
    return result

def crop_rescale(image):
    # original image is 178x218
    # first crop to 178x178
    image = image[20:198,0:178] / 255.0
    # then rescale to 224x224
    image = resize(image, (224, 224), mode='constant')
    # finally, flatten it
    return np.reshape(image, (224*224*3))

