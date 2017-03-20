# 784-1200-1200-10 model
# with 50% dropout on hidden and 20% on visible nodes
import tensorflow as tf

def create_model(inp, out_size, keep_inp=0.8, keep=0.5, temp=1.0):
    with tf.variable_scope('784-1200-1200-10'):
        with tf.variable_scope('inp_drop'):
            inp_drop = tf.nn.dropout(inp, keep_prob=keep_inp)

        with tf.variable_scope('fc1'):
            w = tf.Variable(tf.truncated_normal([int(inp.get_shape()[-1]), 1200]), name='w')
            b = tf.Variable(tf.constant(0.1, shape=[1200]), name='b')
            z = tf.nn.relu(tf.matmul(inp_drop, w) + b, name='relu')
            z_drop = tf.nn.dropout(z, keep_prob=keep)

        with tf.variable_scope('fc2'):
            w = tf.Variable(tf.truncated_normal([1200, 1200]), name='w')
            b = tf.Variable(tf.constant(0.1, shape=[1200]), name='b')
            z = tf.nn.relu(tf.matmul(z_drop, w) + b, name='relu')
            z_drop = tf.nn.dropout(z, keep_prob=keep)

        with tf.variable_scope('fc3'):
            w = tf.Variable(tf.truncated_normal([1200, out_size]), name='w')
            b = tf.Variable(tf.constant(0.1, shape=[out_size]), name='b')
            h = tf.matmul(z_drop, w) + b

        with tf.variable_scope('temp'):
            h_soft = tf.div(h, temp)

    return h_soft
