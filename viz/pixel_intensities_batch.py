import sys
import cv2
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# import datasets as d # see notes below

from random import choice
from view import reshape_to_grid
from view import reshape_to_row

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', '', 'The path to an optimized dataset.')
flags.DEFINE_boolean('rgb', False, 'whether the img is rgb or not')

data = np.load(FLAGS.dataset)[()]
if FLAGS.rgb:
    ex_side = int(np.sqrt(len(data[0][0]) / 3))
else:
    ex_side = int(np.sqrt(len(data[0][0])))

# SANITY CHECK
print('min: {}'.format(min(data[0][0] + 0.5)))
print('max: {}'.format(max(data[0][0] + 0.5)))

# SHOWING ALL EX IN THIS BATCH
cv2.imshow('means.png', reshape_to_grid(np.array(data[0] + 0.5), side=ex_side, rgb=FLAGS.rgb))
cv2.waitKey(0)

# SHOWING MEANS OF THOSE
means = [np.squeeze(np.mean(data[0] + 0.5, axis=0))]
cv2.imshow('means.png', reshape_to_row(np.array(means), side=ex_side, rgb=FLAGS.rgb))
cv2.waitKey(0)

# SANITY CHECK
# ones = [np.ones([ex_side, ex_side, 3])]
# cv2.imshow('means.png', reshape_to_row(np.array(ones), side=ex_side, rgb=FLAGS.rgb))
# cv2.waitKey(0)

