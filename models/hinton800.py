# 784-800-800-10 model
# with no regularization
import tensorflow as tf

def create_model(inp, out_size, keep_inp=1.0, keep=1.0):
    with tf.variable_scope('784-800-800-10'):
        with tf.variable_scope('fc1'):
            w = tf.Variable(tf.truncated_normal([int(inp.get_shape()[-1]), 800]), name='w')
            b = tf.Variable(tf.constant(0.1, shape=[800]), name='b')
            z = tf.nn.relu(tf.matmul(inp, w) + b, name='relu')

        with tf.variable_scope('fc2'):
            w = tf.Variable(tf.truncated_normal([800, 800]), name='w')
            b = tf.Variable(tf.constant(0.1, shape=[800]), name='b')
            z = tf.nn.relu(tf.matmul(z, w) + b, name='relu')

        with tf.variable_scope('fc3'):
            w = tf.Variable(tf.truncated_normal([800, out_size]), name='w')
            b = tf.Variable(tf.constant(0.1, shape=[out_size]), name='b')
            h = tf.matmul(z, w) + b

    return h
