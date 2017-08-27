# alexnet
#
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import flatten
from tensorflow.python.tools import inspect_checkpoint

def create_model(inputs, output_size):
    imagenet_init = False
    imnet = None if not imagenet_init else Exception('You never loaded imagenet weights. Not implemented!')

    layer_activations = []

    with tf.variable_scope('alex'):
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        temperature = tf.placeholder(tf.float32, name='temperature')

        # MAKE SURE INPUTS ARE 224x224 IMAGES, otherwise the layer size below is wrong!!!
        inputs_reshaped = tf.reshape(inputs, [-1, 224, 224, 3])

        with tf.variable_scope('conv_pool_1'):
            conv1_1 = convLayer(inputs_reshaped, 11, 11, 4, 4, 64, "conv1_1", layer_activations, 56*56*64, init_dict=imnet)
            pool1 = maxPoolLayer(conv1_1, 3, 3, 2, 2, "pool1")

        with tf.variable_scope('conv_pool_2'):
            conv2_1 = convLayer(pool1, 5, 5, 1, 1, 192, "conv2_1", layer_activations, 28*28*192, init_dict=imnet)
            pool2 = maxPoolLayer(conv2_1, 3, 3, 2, 2, "pool2")

        with tf.variable_scope('conv_pool_3'):
            conv3_1 = convLayer(pool2, 3, 3, 1, 1, 384, "conv3_1", layer_activations, 14*14*384, init_dict=imnet)

        with tf.variable_scope('conv_pool_4'):
            conv4_1 = convLayer(conv3_1, 3, 3, 1, 1, 384, "conv4_1", layer_activations, 14*14*384, init_dict=imnet)

        with tf.variable_scope('conv_pool_5'):
            conv5_1 = convLayer(conv4_1, 3, 3, 1, 1, 256, "conv5_1", layer_activations, 14*14*256, init_dict=imnet)
            pool5 = maxPoolLayer(conv5_1, 3, 3, 2, 2, "pool5")

        with tf.variable_scope('fc_6'):
            fcIn = tf.reshape(pool5, [-1, 7*7*256])
            fc6 = fcLayer(fcIn, 7*7*256, 4096, True, "fc6", layer_activations, init_dict=imnet)
            dropout1 = dropout(fc6, keep_prob)

        with tf.variable_scope('fc_7'):
            fc7 = fcLayer(dropout1, 4096, 4096, True, "fc7", layer_activations, init_dict=imnet)
            dropout2 = dropout(fc7, keep_prob)

        with tf.variable_scope('fc_8'):
            # need to do temperature here, so can't call the helper function
            with tf.variable_scope("fc8") as scope:
                w = tf.Variable(
                        tf.truncated_normal(shape=[4096, output_size], stddev=np.sqrt(2.0/4096)),
                        name='fc8_w')
                        #  tf.truncated_normal(shape=[4096, output_size], stddev=0.01),
                        #  name='fc8_w')
                b = tf.Variable(tf.constant(0.01, shape=[output_size]), name='fc8_b')
                #  b = tf.Variable(tf.constant(0.0, shape=[output_size]), name='fc8_b')

                out = tf.matmul(dropout2, w) + b

                tf.add_to_collection('fc8_w', w)
                tf.add_to_collection('fc8_b', b)

            with tf.variable_scope("temperature"):
                out_soft = tf.div(out, temperature)
                layer_activations.append((out_soft, output_size))

    tf.add_to_collection('inputs', inputs)
    tf.add_to_collection('outputs', out_soft)
    tf.add_to_collection('keep_prob', keep_prob)
    tf.add_to_collection('temperature', temperature)

    feed_dicts = create_feed_dicts(keep_prob, temperature)

    return out_soft, layer_activations, feed_dicts


### HELPER FUNCTIONS START ###
def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding="SAME"):
    return tf.nn.max_pool(x, ksize=[1, kHeight, kWidth, 1],
            strides=[1, strideX, strideY, 1], padding=padding, name=name)

def dropout(x, keepPro, name=None):
    return tf.nn.dropout(x, keepPro, name)

def fcLayer(x, inputD, outputD, reluFlag, name, layer_activations, init_dict=None):
    with tf.variable_scope(name) as scope:
        if init_dict is None:
            w = tf.Variable(
                    tf.truncated_normal(shape=[inputD, outputD], stddev=np.sqrt(2.0/inputD)),
                    name='{}_w'.format(name))
                    #  tf.truncated_normal(shape=[inputD, outputD], stddev=0.01),
                    #  name='{}_w'.format(name))
            b = tf.Variable(tf.constant(0.01, shape=[outputD]), name='{}_b'.format(name))
            #  b = tf.Variable(tf.constant(0.0, shape=[outputD]), name='{}_b'.format(name))
        else:
            w = tf.Variable(tf.constant(init_dict[name][0]), name='{}_w'.format(name))
            b = tf.Variable(tf.constant(init_dict[name][1]), name='{}_b'.format(name))

        out = tf.matmul(x, w) + b

        tf.add_to_collection('{}_w'.format(name), w)
        tf.add_to_collection('{}_b'.format(name), b)
        tf.add_to_collection(name, out)
        layer_activations.append((out, outputD))

        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out

def convLayer(x, kHeight, kWidth, strideX, strideY,
              featureNum, name, layer_activations, out_size, init_dict=None, padding="SAME"):
    channel = int(x.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        if init_dict is None:
            w = tf.Variable(
                    tf.truncated_normal([kHeight, kWidth, channel, featureNum], stddev=np.sqrt(2.0/(kHeight*kWidth*channel))),
                    name='{}_w'.format(name))
                    #  tf.truncated_normal([kHeight, kWidth, channel, featureNum], stddev=0.01),
                    #  name='{}_w'.format(name))
            b = tf.Variable(tf.constant(0.01, shape=[featureNum]), name='{}_b'.format(name))
            #  b = tf.Variable(tf.constant(0.0, shape=[featureNum]), name='{}_b'.format(name))
        else:
            w = tf.Variable(tf.constant(init_dict[name][0]), name='{}_w'.format(name))
            b = tf.Variable(tf.constant(init_dict[name][1]), name='{}_b'.format(name))
        out = tf.nn.conv2d(x, w, strides=[1, strideY, strideX, 1], padding=padding) + b

        tf.add_to_collection('{}_w'.format(name), w)
        tf.add_to_collection('{}_b'.format(name), b)
        tf.add_to_collection(name, out)
        layer_activations.append((out, out_size))

        return tf.nn.relu(out)
### HELPER FUNCTIONS END ###


def load_model(sess, model_meta, model_checkpoint, output_size):
    new_saver = tf.train.import_meta_graph(model_meta)
    new_saver.restore(sess, model_checkpoint)

    inputs = tf.get_collection('inputs')[0]
    outputs = tf.get_collection('outputs')[0]
    keep_prob = tf.get_collection('keep_prob')[0]
    temperature = tf.get_collection('temperature')[0]

    layer_activations = []
    #  layer_activations.append((tf.get_collection('conv1_1')[0], 56*56*64))
    #  layer_activations.append((tf.get_collection('conv2_1')[0], 28*28*192))
    #  layer_activations.append((tf.get_collection('conv3_1')[0], 14*14*384))
    #  layer_activations.append((tf.get_collection('conv4_1')[0], 14*14*384))
    #  layer_activations.append((tf.get_collection('conv5_1')[0], 14*14*256))
    #  layer_activations.append((tf.get_collection('fc6')[0], 4096))
    #  layer_activations.append((tf.get_collection('fc7')[0], 4096))
    layer_activations.append((outputs, output_size))

    feed_dicts = create_feed_dicts(keep_prob, temperature)

    return inputs, outputs, layer_activations, feed_dicts


def load_and_freeze_model(sess, inputs, model_meta, model_checkpoint, batch_size, output_size):
    new_saver = tf.train.import_meta_graph(model_meta)
    new_saver.restore(sess, model_checkpoint)

    layer_activations = []

    with tf.variable_scope('alex'):
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        temperature = tf.placeholder(tf.float32, name='temperature')

        # MAKE SURE INPUTS ARE 224x224 IMAGES, otherwise the layer size below is wrong!!!
        inputs_reshaped = tf.reshape(inputs, [-1, 224, 224, 3])

        with tf.variable_scope('conv_pool_1'):
            conv1_1_w = tf.constant(sess.run(tf.get_collection('conv1_1_w')[0]), name='conv1_1_w')
            conv1_1_b = tf.constant(sess.run(tf.get_collection('conv1_1_b')[0]), name='conv1_1_b')
            conv1_1   = tf.nn.conv2d(inputs_reshaped, conv1_1_w, strides=[1,4,4,1], padding='SAME') + conv1_1_b

            pool1 = maxPoolLayer(conv1_1, 3,3,2,2, name="pool1")

            #  layer_activations.append((conv1_1, 56*56*64))

        with tf.variable_scope('conv_pool_2'):
            conv2_1_w = tf.constant(sess.run(tf.get_collection('conv2_1_w')[0]), name='conv2_1_w')
            conv2_1_b = tf.constant(sess.run(tf.get_collection('conv2_1_b')[0]), name='conv2_1_b')
            conv2_1   = tf.nn.conv2d(pool1, conv2_1_w, strides=[1,1,1,1], padding='SAME') + conv2_1_b

            pool2 = maxPoolLayer(conv2_1, 3,3,2,2,  "pool2")

            #  layer_activations.append((conv2_1, 28*28*192))

        with tf.variable_scope('conv_pool_3'):

            conv3_1_w = tf.constant(sess.run(tf.get_collection('conv3_1_w')[0]), name='conv3_1_w')
            conv3_1_b = tf.constant(sess.run(tf.get_collection('conv3_1_b')[0]), name='conv3_1_b')
            conv3_1   = tf.nn.conv2d(pool2, conv3_1_w, strides=[1,1,1,1], padding='SAME') + conv3_1_b

            #  layer_activations.append((conv3_1, 14*14*384))

        with tf.variable_scope('conv_pool_4'):
            conv4_1_w = tf.constant(sess.run(tf.get_collection('conv4_1_w')[0]), name='conv4_1_w')
            conv4_1_b = tf.constant(sess.run(tf.get_collection('conv4_1_b')[0]), name='conv4_1_b')
            conv4_1   = tf.nn.conv2d(conv3_1, conv4_1_w, strides=[1,1,1,1], padding='SAME') + conv4_1_b

            #  layer_activations.append((conv4_1, 14*14*384))

        with tf.variable_scope('conv_pool_5'):
            conv5_1_w = tf.constant(sess.run(tf.get_collection('conv5_1_w')[0]), name='conv5_1_w')
            conv5_1_b = tf.constant(sess.run(tf.get_collection('conv5_1_b')[0]), name='conv5_1_b')
            conv5_1   = tf.nn.conv2d(conv4_1, conv5_1_w, strides=[1,1,1,1], padding='SAME') + conv5_1_b

            pool5 = maxPoolLayer(conv5_1, 3, 3, 2, 2, "pool5")

            #  layer_activations.append((conv5_1, 14*14*256))

        with tf.variable_scope('fc_6'):
            fcIn = tf.reshape(pool5, [-1, 7*7*256])
            fc6_w = tf.constant(sess.run(tf.get_collection('fc6_w')[0]), name='fc6_w')
            fc6_b = tf.constant(sess.run(tf.get_collection('fc6_b')[0]), name='fc6_b')
            fc6 = tf.matmul(fcIn, fc6_w) + fc6_b

            #  layer_activations.append((fc6, 4096))

            dropout1 = dropout(fc6, keep_prob)

        with tf.variable_scope('fc_7'):

            fc7_w = tf.constant(sess.run(tf.get_collection('fc7_w')[0]), name='fc7_w')
            fc7_b = tf.constant(sess.run(tf.get_collection('fc7_b')[0]), name='fc7_b')
            fc7 = tf.matmul(dropout1, fc7_w) + fc7_b

            #  layer_activations.append((fc7, 4096))

            dropout2 = dropout(fc7, keep_prob)

        with tf.variable_scope('fc_8'):

            fc8_w = tf.constant(sess.run(tf.get_collection('fc8_w')[0]), name='fc8_w')
            fc8_b = tf.constant(sess.run(tf.get_collection('fc8_b')[0]), name='fc8_b')
            logits = tf.matmul(dropout2, fc8_w) + fc8_b
            logits = tf.div(logits, temperature)

    layer_activations.append((logits, output_size))

    feed_dicts = create_feed_dicts(keep_prob, temperature)
    return logits, layer_activations, feed_dicts, []

def create_feed_dicts(keep_prob, temperature):
    feed_dicts = {key: {} for key in ['train', 'eval', 'distill']}

    feed_dicts['train'][temperature] = 1.0
    feed_dicts['train'][keep_prob] = 0.5

    feed_dicts['eval'][temperature] = 1.0
    feed_dicts['eval'][keep_prob] = 1.0

    feed_dicts['distill'][temperature] = 8.0
    feed_dicts['distill'][keep_prob] = 1.0

    return feed_dicts
