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
    with tf.variable_scope('784-1200-1200-10_const'):
        with tf.variable_scope('inp_drop'):
            # mere rescaling
            #  inp = inp * 0.8
            # TODO NIPS: use drop_dict
            inp = tf.multiply(inp, drop_dict['drop_inp_var'])

        with tf.variable_scope('fc1'):
            w = tf.constant(sess.run('784-1200-1200-10/fc1/w:0'), name='w')
            b = tf.constant(sess.run('784-1200-1200-10/fc1/b:0'), name='b')
            z = tf.nn.relu(tf.matmul(inp, w) + b, name='relu')
            # mere rescaling
            #  z = z * 0.5
            # TODO NIPS: use drop_dict
            z = tf.multiply(z, drop_dict['drop_fc1_var'])

        with tf.variable_scope('fc2'):
            w = tf.constant(sess.run('784-1200-1200-10/fc2/w:0'), name='w')
            b = tf.constant(sess.run('784-1200-1200-10/fc2/b:0'), name='b')
            z = tf.nn.relu(tf.matmul(z, w) + b, name='relu')
            #  z = z * 0.5
            # TODO NIPS: use drop_dict
            z = tf.multiply(z, drop_dict['drop_fc2_var'])

        with tf.variable_scope('fc3'):
            w = tf.constant(sess.run('784-1200-1200-10/fc3/w:0'), name='w')
            b = tf.constant(sess.run('784-1200-1200-10/fc3/b:0'), name='b')
            h = tf.matmul(z, w) + b

    return h

