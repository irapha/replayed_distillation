# lenet-5 model
#
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import flatten

def create_model(inp, out_size, temp=1.0):
    with tf.variable_scope('lenet-5'):
        with tf.variable_scope('conv1'):
            # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
            conv1_w = tf.Variable(tf.truncated_normal([5,5,1,6], stddev=0.1), name='conv1_w')
            conv1_b = tf.Variable(tf.zeros(6), name='conv1_b')
            conv1 = tf.nn.conv2d(inp, conv1_w, strides=[1,1,1,1], padding='VALID') + conv1_b
            # Activation.
            conv1 = tf.nn.relu(conv1)

        with tf.variable_scope('pool1'):
            # Pooling. Input = 28x28x6. Output = 14x14x6.
            pool_1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

        with tf.variable_scope('conv2'):
            # Layer 2: Convolutional. Output = 10x10x16.
            conv2_w = tf.Variable(tf.truncated_normal(shape=[5,5,6,16], stddev=0.1), name='conv2_w')
            conv2_b = tf.Variable(tf.zeros(16), name='conv2_b')
            conv2 = tf.nn.conv2d(pool_1, conv2_w, strides=[1,1,1,1], padding='VALID') + conv2_b
            # Activation.
            conv2 = tf.nn.relu(conv2)

        with tf.variable_scope('pool2'):
            # Pooling. Input = 10x10x16. Output = 5x5x16.
            pool_2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

        with tf.variable_scope('flatten'):
            # Flatten. Input = 5x5x16. Output = 400.
            fc1 = flatten(pool_2)

        with tf.variable_scope('fc1'):
            # Layer 3: Fully Connected. Input = 400. Output = 120.
            fc1_w = tf.Variable(tf.truncated_normal(shape=(400,120), stddev=0.1), name='fc1_w')
            fc1_b = tf.Variable(tf.zeros(120), name='fc1_b')
            fc1 = tf.matmul(fc1,fc1_w) + fc1_b
            # Activation.
            fc1 = tf.nn.relu(fc1)

        with tf.variable_scope('fc2'):
            # Layer 4: Fully Connected. Input = 120. Output = 84.
            fc2_w = tf.Variable(tf.truncated_normal(shape=(120,84), stddev=0.1), name='fc2_w')
            fc2_b = tf.Variable(tf.zeros(84), name='fc2_b')
            fc2 = tf.matmul(fc1,fc2_w) + fc2_b
            # Activation.
            fc2 = tf.nn.relu(fc2)

        with tf.variable_scope('fc3'):
            # Layer 5: Fully Connected. Input = 84. Output = 10.
            fc3_w = tf.Variable(tf.truncated_normal(shape=(84,10), stddev=0.1), name='fc3_w')
            fc3_b = tf.Variable(tf.zeros(10), name='fc3_b')
            logits = tf.matmul(fc2, fc3_w) + fc3_b

        with tf.variable_scope('temp'):
            logits_soft = tf.div(logits, temp)

    return logits_soft


def create_constant_model(sess, inp, drop_dict):
    with tf.variable_scope('lenet-5_const'):
        with tf.variable_scope('conv1'):
            conv1_w = tf.constant(sess.run('lenet-5/conv1/conv1_w:0'), name='conv1_w')
            conv1_b = tf.constant(sess.run('lenet-5/conv1/conv1_b:0'), name='conv1_b')
            conv1 = tf.nn.conv2d(inp, conv1_w, strides=[1,1,1,1], padding='VALID') + conv1_b
            conv1 = tf.nn.relu(conv1)

        with tf.variable_scope('pool1'):
            pool_1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

        with tf.variable_scope('conv2'):
            conv2_w = tf.constant(sess.run('lenet-5/conv2/conv2_w:0'), name='conv2_w')
            conv2_b = tf.constant(sess.run('lenet-5/conv2/conv2_b:0'), name='conv2_b')
            conv2 = tf.nn.conv2d(pool_1, conv2_w, strides=[1,1,1,1], padding='VALID') + conv2_b
            conv2 = tf.nn.relu(conv2)

        with tf.variable_scope('pool2'):
            pool_2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

        with tf.variable_scope('flatten'):
            fc1 = flatten(pool_2)

        with tf.variable_scope('fc1'):
            fc1_w = tf.constant(sess.run('lenet-5/fc1/fc1_w:0'), name='fc1_w')
            fc1_b = tf.constant(sess.run('lenet-5/fc1/fc1_b:0'), name='fc1_b')
            fc1 = tf.matmul(fc1,fc1_w) + fc1_b
            fc1 = tf.nn.relu(fc1)

        with tf.variable_scope('fc2'):
            fc2_w = tf.constant(sess.run('lenet-5/fc2/fc2_w:0'), name='fc2_w')
            fc2_b = tf.constant(sess.run('lenet-5/fc2/fc2_b:0'), name='fc2_b')
            fc2 = tf.matmul(fc1,fc2_w) + fc2_b
            fc2 = tf.nn.relu(fc2)

        with tf.variable_scope('fc3'):
            fc3_w = tf.constant(sess.run('lenet-5/fc3/fc3_w:0'), name='fc3_w')
            fc3_b = tf.constant(sess.run('lenet-5/fc3/fc3_b:0'), name='fc3_b')
            logits = tf.matmul(fc2, fc3_w) + fc3_b

    return logits

