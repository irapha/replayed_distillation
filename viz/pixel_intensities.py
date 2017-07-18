import sys
import cv2
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# import datasets as d # see notes below

from random import choice
from view import reshape_to_row

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', '', 'The path to an optimized dataset.')


data = np.load(FLAGS.dataset)[()]
num_classes = len(data[0][0][1][0][0])
# ^ see datasets/optimized_dataset.py:OptimizedDataset.io_size

print('computing per-class, per-pixel means')
means = []
for clas in range(num_classes):
    clas_imgs = []
    for batch_x, _ in data[clas]:
        clas_imgs.extend(batch_x)
    means.append(np.squeeze(np.mean(clas_imgs, axis=0)))
cv2.imshow('means.png', reshape_to_row(np.array(means)))
# cv2.imwrite('MEANS_{}.png'.format(dat[:-4]), 255*reshape_to_row(np.array(means), 28))

print('picking a random example per class')
random = []
for clas in range(10):
    clas_imgs = []
    for batch_x, _ in data[clas]:
        clas_imgs.extend(batch_x)
    random.append(choice(clas_imgs))
cv2.imshow('random.png', reshape_to_row(np.array(random)))
# cv2.imwrite('RAND_{}.png'.format(dat[:-4]), 255*reshape_to_row(np.array(random), 28))

cv2.waitKey(0)
cv2.destroyAllWindows()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# The below was used to create per-class histograms for both the original
# dataset and the reconstructed one. This way, we could get an idea of how
# closely the reconstruction matched the original dataset.
#
# It is currently broken due to `d` not being available from within the viz/
# folder. I won't delete it because someone (you) might find it useful.
#
# If you want to fix this code and run it, you may run the following:
# `$ cp -r datasets viz/`
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# if not FLAGS.original_dataset:
    # print('no original dataset provided, cannot analyze brightness and contrast for comparison')
    # sys.exit(0)

# re_means = means # calculated above

# mnist = d.get(FLAGS.original_dataset)
# og_means = []
# for clas in range(10):
    # idx = np.where(np.where(mnist.og.train.labels == 1)[1] == clas)[0]
    # og_means.append(np.mean(mnist.og.train.images[idx], axis=0))
    # og_random.append(choice(mnist.og.train.images[idx]))

# print('analysing brightness and contrast')
# f, subs = plt.subplots(1, 10, sharey=True, sharex=True, figsize=(40, 3))
# x = np.linspace(0.0, 1.0, 30)
# plt.locator_params(axis='y', nbins=3)
# plt.locator_params(axis='x', nbins=2)

# for i, (og, re) in enumerate(zip(og_means, re_means)):
    # grey = '#666666'
    # light_grey = '#b7b7b7'
    # subs[i].tick_params(axis='x', colors=grey)
    # subs[i].tick_params(axis='y', colors=grey)
    # subs[i].spines['bottom'].set_color(grey)
    # subs[i].spines['top'].set_color(grey)
    # subs[i].spines['left'].set_color(grey)
    # subs[i].spines['right'].set_color(grey)

    # subs[i].hist(og, x, facecolor='#a64d79', alpha=0.5)
    # subs[i].hist(re, x, facecolor='#674ea7', alpha=0.5)

    # # subs[i].plot(x, mlab.normpdf(x, np.mean(og), np.sqrt(np.var(og))), 'r')
    # # subs[i].plot(x, mlab.normpdf(x, np.mean(re), np.sqrt(np.var(re))), 'b')

# print('og is red, re is blue')
# plt.show()
