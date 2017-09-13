# lenet-5 model
#
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import flatten

def create_model(inputs, output_size):
    layer_activations = []

    with tf.variable_scope('lenet-5'):
        temperature = tf.placeholder(tf.float32, name='temperature')

        # MAKE SURE INPUTS ARE 32x32 IMAGES, otherwise the layer size below is wrong!!!
        inputs_reshaped = tf.reshape(inputs, [-1, 32, 32, 1])

        with tf.variable_scope('conv1'):
            # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
            conv1_w = tf.Variable(tf.truncated_normal([5,5,1,6], stddev=0.1), name='conv1_w')
            tf.add_to_collection('conv1_w', conv1_w)
            conv1_b = tf.Variable(tf.zeros(6), name='conv1_b')
            tf.add_to_collection('conv1_b', conv1_b)
            conv1 = tf.nn.conv2d(inputs_reshaped, conv1_w, strides=[1,1,1,1], padding='VALID') + conv1_b

            # MAKE SURE INPUTS ARE 32x32 IMAGES, otherwise the layer size below is wrong!!!
            layer_activations.append((conv1, 28*28*6))
            tf.add_to_collection('conv1', conv1)

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

            layer_activations.append((conv2, 10*10*16))
            tf.add_to_collection('conv2_w', conv2_w)
            tf.add_to_collection('conv2_b', conv2_b)
            tf.add_to_collection('conv2', conv2)

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

            layer_activations.append((fc1, 120))
            tf.add_to_collection('fc1_w', fc1_w)
            tf.add_to_collection('fc1_b', fc1_b)
            tf.add_to_collection('fc1', fc1)

            # Activation.
            fc1 = tf.nn.relu(fc1)

        with tf.variable_scope('fc2'):
            # Layer 4: Fully Connected. Input = 120. Output = 84.
            fc2_w = tf.Variable(tf.truncated_normal(shape=(120,84), stddev=0.1), name='fc2_w')
            fc2_b = tf.Variable(tf.zeros(84), name='fc2_b')
            fc2 = tf.matmul(fc1,fc2_w) + fc2_b

            layer_activations.append((fc2, 84))
            tf.add_to_collection('fc2_w', fc2_w)
            tf.add_to_collection('fc2_b', fc2_b)
            tf.add_to_collection('fc2', fc2)

            # Activation.
            fc2 = tf.nn.relu(fc2)

        with tf.variable_scope('fc3'):
            # Layer 5: Fully Connected. Input = 84. Output = 10.
            fc3_w = tf.Variable(tf.truncated_normal(shape=(84,output_size), stddev=0.1), name='fc3_w')
            fc3_b = tf.Variable(tf.zeros(output_size), name='fc3_b')
            logits = tf.matmul(fc2, fc3_w) + fc3_b

            tf.add_to_collection('fc3_w', fc3_w)
            tf.add_to_collection('fc3_b', fc3_b)

        with tf.variable_scope('temp'):
            logits_soft = tf.div(logits, temperature)

            layer_activations.append((logits_soft, output_size))

    tf.add_to_collection('inputs', inputs)
    tf.add_to_collection('outputs', logits_soft)
    tf.add_to_collection('temperature', temperature)

    feed_dicts = create_feed_dicts(temperature)

    return logits_soft, layer_activations, feed_dicts

def load_model(sess, model_meta, model_checkpoint, output_size):
    new_saver = tf.train.import_meta_graph(model_meta)
    new_saver.restore(sess, model_checkpoint)

    inputs = tf.get_collection('inputs')[0]
    outputs = tf.get_collection('outputs')[0]
    temperature = tf.get_collection('temperature')[0]

    layer_activations = []
    layer_activations.append((tf.get_collection('conv1')[0], 28*28*6))
    layer_activations.append((tf.get_collection('conv2')[0], 10*10*16))
    layer_activations.append((tf.get_collection('fc1')[0], 120))
    layer_activations.append((tf.get_collection('fc2')[0], 84))
    #  layer_activations.append((outputs, int(outputs.get_shape()[-1])))
    # the above doesn't work because tensorflow 1.0 has a bug where restored
    # variables have get_shape == <unknown>. So we just take the output_size
    # from dataset. It's messier but it works.
    layer_activations.append((outputs, output_size))

    feed_dicts = create_feed_dicts(temperature)

    return inputs, outputs, layer_activations, feed_dicts

def load_and_freeze_model(sess, inputs, model_meta, model_checkpoint, batch_size, output_size):
    new_saver = tf.train.import_meta_graph(model_meta)
    new_saver.restore(sess, model_checkpoint)
    temperature = tf.placeholder(tf.float32, name='temperature')

    layer_activations = []

    with tf.variable_scope('lenet-5_const'):

        # MAKE SURE INPUTS ARE 32x32 IMAGES, otherwise the layer size below is wrong!!!
        inputs_reshaped = tf.reshape(inputs, [-1, 32, 32, 1])

        with tf.variable_scope('conv1'):
            conv1_w = tf.constant(sess.run(tf.get_collection('conv1_w')[0]), name='conv1_w')
            conv1_b = tf.constant(sess.run(tf.get_collection('conv1_b')[0]), name='conv1_b')
            conv1 = tf.nn.conv2d(inputs_reshaped, conv1_w, strides=[1,1,1,1], padding='VALID') + conv1_b

            layer_activations.append((conv1, 28*28*6))

            conv1 = tf.nn.relu(conv1)

        with tf.variable_scope('pool1'):
            pool_1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

        with tf.variable_scope('conv2'):
            conv2_w = tf.constant(sess.run(tf.get_collection('conv2_w')[0]), name='conv2_w')
            conv2_b = tf.constant(sess.run(tf.get_collection('conv2_b')[0]), name='conv2_b')
            conv2 = tf.nn.conv2d(pool_1, conv2_w, strides=[1,1,1,1], padding='VALID') + conv2_b

            layer_activations.append((conv2, 10*10*16))

            conv2 = tf.nn.relu(conv2)

        with tf.variable_scope('pool2'):
            pool_2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

        with tf.variable_scope('flatten'):
            fc1 = flatten(pool_2)

        with tf.variable_scope('fc1'):
            fc1_w = tf.constant(sess.run(tf.get_collection('fc1_w')[0]), name='fc1_w')
            fc1_b = tf.constant(sess.run(tf.get_collection('fc1_b')[0]), name='fc1_b')
            fc1 = tf.matmul(fc1,fc1_w) + fc1_b

            layer_activations.append((fc1, 120))

            fc1 = tf.nn.relu(fc1)

        with tf.variable_scope('fc2'):
            fc2_w = tf.constant(sess.run(tf.get_collection('fc2_w')[0]), name='fc2_w')
            fc2_b = tf.constant(sess.run(tf.get_collection('fc2_b')[0]), name='fc2_b')
            fc2 = tf.matmul(fc1,fc2_w) + fc2_b

            layer_activations.append((fc2, 84))

            fc2 = tf.nn.relu(fc2)

        with tf.variable_scope('fc3'):
            fc3_w = tf.constant(sess.run(tf.get_collection('fc3_w')[0]), name='fc3_w')
            fc3_b = tf.constant(sess.run(tf.get_collection('fc3_b')[0]), name='fc3_b')
            logits = tf.matmul(fc2, fc3_w) + fc3_b
            logits = tf.div(logits, temperature)

            layer_activations.append((logits, 10))

    feed_dicts = {'distill': {temperature: 8.0}}

    return logits, layer_activations, feed_dicts, []

def create_feed_dicts(temperature):
    feed_dicts = {key: {} for key in ['train', 'eval', 'distill']}

    feed_dicts['train'][temperature] = 1.0
    feed_dicts['eval'][temperature] = 1.0
    feed_dicts['distill'][temperature] = 8.0

    return feed_dicts
