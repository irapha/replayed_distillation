import numpy as np
import datasets as d
import cv2

from viz.view import reshape_to_row, reshape_to_grid

# INIT CELEBA
#  celeba = d.get('celeba', [])
#  batch, _ = next(celeba.train_epoch_in_batches(64))
#  print(np.shape(list(batch)))
#  ex_side = int(np.sqrt(len(batch[0]) / 3))

# SANITY CHECK
#  print('min: {}'.format(min(batch[0])))
#  print('max: {}'.format(max(batch[0])))
#  print('mean: {}'.format(np.mean(np.reshape(batch, (-1,)))))
#  print('std: {}'.format(np.std(np.reshape(batch, (-1,)))))

# SHOWING ALL EX IN THIS BATCH
#  cv2.imshow('normal.png', reshape_to_grid(np.array(batch), side=ex_side, rgb=True))
#  cv2.waitKey(0)

# SHOWING MEANS OF THOSE
#  means = [np.squeeze(np.mean(batch, axis=0))]
#  cv2.imshow('means.png', reshape_to_row(np.array(means), side=ex_side, rgb=True))
#  cv2.waitKey(0)


# TESTING HOW TO BLUR IN TF
#  import tensorflow as tf
#  blur_kernel = tf.constant([
    #  [[[1/16, 0, 0], [0, 1/16, 0], [0, 0, 1/16]], [[1/8, 0, 0], [0, 1/8, 0], [0, 0, 1/8]], [[1/16, 0 ,0], [0, 1/16, 0], [0, 0, 1/16]]],
    #  [[[1/8 , 0, 0], [0, 1/8 , 0], [0, 0, 1/8 ]], [[1/4, 0, 0], [0, 1/4, 0], [0, 0, 1/4]], [[1/8 , 0 ,0], [0, 1/8 , 0], [0, 0, 1/8 ]]],
    #  [[[1/16, 0, 0], [0, 1/16, 0], [0, 0, 1/16]], [[1/8, 0, 0], [0, 1/8, 0], [0, 0, 1/8]], [[1/16, 0 ,0], [0, 1/16, 0], [0, 0, 1/16]]]])
#  imgs = tf.constant(np.asarray(batch), dtype=tf.float32)


#  sess = tf.Session()
#  blurred = sess.run(tf.nn.conv2d(tf.reshape(imgs, [64, 224, 224, 3]), blur_kernel, strides=[1,1,1,1], padding='SAME'))

#  cv2.imshow('normal.png', reshape_to_grid(np.array(batch), side=ex_side, rgb=True))
#  cv2.waitKey(10000)
#  cv2.imshow('blurred.png', reshape_to_grid(np.array(blurred), side=ex_side, rgb=True))
#  cv2.waitKey(10000)


# INIT MNIST
mnist = d.get('mnist', None)
batch, _ = next(mnist.train_epoch_in_batches(64))
print(np.shape(list(batch)))
ex_side = int(np.sqrt(len(batch[0])))
print(ex_side)


# TESTING BLUR IN TF FOR MNIST
import tensorflow as tf
blur_kernel= tf.constant([
    [[[1/16]], [[1/8 ]], [[1/16]]],
    [[[1/8 ]], [[1/4 ]], [[1/8 ]]],
    [[[1/16]], [[1/8 ]], [[1/16]]]])
imgs = tf.constant(np.asarray(batch), dtype=tf.float32)


sess = tf.Session()
blurred = sess.run(tf.nn.conv2d(tf.reshape(imgs, [64, 28, 28, 1]), blur_kernel, strides=[1,1,1,1], padding='SAME'))

cv2.imshow('normal.png', reshape_to_grid(np.array(batch), side=ex_side, rgb=False))
cv2.waitKey(10000)
cv2.imshow('blurred.png', reshape_to_grid(np.array(blurred), side=ex_side, rgb=False))
cv2.waitKey(10000)
