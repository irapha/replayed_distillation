# vgg 19
#
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import flatten
from tensorflow.python.tools import inspect_checkpoint

def create_model(inputs, output_size):
    layer_activations = []

    with tf.variable_scope('vgg'):
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        temperature = tf.placeholder(tf.float32, name='temperature')

        # MAKE SURE INPUTS ARE 224x224 IMAGES, otherwise the layer size below is wrong!!!
        inputs_reshaped = tf.reshape(inputs, [-1, 224, 224, 1])

        with tf.variable_scope('conv_pool_1'):
            conv1_1 = convLayer(inputs_reshaped, 3, 3, 1, 1, 64, "conv1_1", layer_activations, 224*224*64)
            conv1_2 = convLayer(conv1_1, 3, 3, 1, 1, 64, "conv1_2", layer_activations, 224*224*64)
            pool1 = maxPoolLayer(conv1_2, 2, 2, 2, 2, "pool1")

        with tf.variable_scope('conv_pool_2'):
            conv2_1 = convLayer(pool1, 3, 3, 1, 1, 128, "conv2_1", layer_activations, 112*112*128)
            conv2_2 = convLayer(conv2_1, 3, 3, 1, 1, 128, "conv2_2", layer_activations, 112*112*128)
            pool2 = maxPoolLayer(conv2_2, 2, 2, 2, 2, "pool2")

        with tf.variable_scope('conv_pool_3'):
            conv3_1 = convLayer(pool2, 3, 3, 1, 1, 256, "conv3_1", layer_activations, 56*56*256)
            conv3_2 = convLayer(conv3_1, 3, 3, 1, 1, 256, "conv3_2", layer_activations, 56*56*256)
            conv3_3 = convLayer(conv3_2, 3, 3, 1, 1, 256, "conv3_3", layer_activations, 56*56*256)
            conv3_4 = convLayer(conv3_3, 3, 3, 1, 1, 256, "conv3_4", layer_activations, 56*56*256)
            pool3 = maxPoolLayer(conv3_4, 2, 2, 2, 2, "pool3")

        with tf.variable_scope('conv_pool_4'):
            conv4_1 = convLayer(pool3, 3, 3, 1, 1, 512, "conv4_1", layer_activations, 28*28*512)
            conv4_2 = convLayer(conv4_1, 3, 3, 1, 1, 512, "conv4_2", layer_activations, 28*28*512)
            conv4_3 = convLayer(conv4_2, 3, 3, 1, 1, 512, "conv4_3", layer_activations, 28*28*512)
            conv4_4 = convLayer(conv4_3, 3, 3, 1, 1, 512, "conv4_4", layer_activations, 28*28*512)
            pool4 = maxPoolLayer(conv4_4, 2, 2, 2, 2, "pool4")

        with tf.variable_scope('conv_pool_5'):
            conv5_1 = convLayer(pool4, 3, 3, 1, 1, 512, "conv5_1", layer_activations, 14*14*512)
            conv5_2 = convLayer(conv5_1, 3, 3, 1, 1, 512, "conv5_2", layer_activations, 14*14*512)
            conv5_3 = convLayer(conv5_2, 3, 3, 1, 1, 512, "conv5_3", layer_activations, 14*14*512)
            conv5_4 = convLayer(conv5_3, 3, 3, 1, 1, 512, "conv5_4", layer_activations, 14*14*512)
            pool5 = maxPoolLayer(conv5_4, 2, 2, 2, 2, "pool5")

        with tf.variable_scope('fc_6'):
            fcIn = tf.reshape(pool5, [-1, 7*7*512])
            fc6 = fcLayer(fcIn, 7*7*512, 4096, True, "fc6", layer_activations)
            dropout1 = dropout(fc6, keep_prob)

        with tf.variable_scope('fc_7'):
            fc7 = fcLayer(dropout1, 4096, 4096, True, "fc7", layer_activations)
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

def fcLayer(x, inputD, outputD, reluFlag, name, layer_activations):
    with tf.variable_scope(name) as scope:
        w = tf.Variable(
                tf.truncated_normal(shape=[inputD, outputD], stddev=np.sqrt(2.0/inputD)),
                name='{}_w'.format(name))
                #  tf.truncated_normal(shape=[inputD, outputD], stddev=0.01),
                #  name='{}_w'.format(name))
        b = tf.Variable(tf.constant(0.01, shape=[outputD]), name='{}_b'.format(name))
        #  b = tf.Variable(tf.constant(0.0, shape=[outputD]), name='{}_b'.format(name))
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
              featureNum, name, layer_activations, out_size, padding="SAME"):
    channel = int(x.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        w = tf.Variable(
                tf.truncated_normal([kHeight, kWidth, channel, featureNum], stddev=np.sqrt(2.0/(kHeight*kWidth*channel))),
                name='{}_w'.format(name))
                #  tf.truncated_normal([kHeight, kWidth, channel, featureNum], stddev=0.01),
                #  name='{}_w'.format(name))
        b = tf.Variable(tf.constant(0.01, shape=[featureNum]), name='{}_b'.format(name))
        #  b = tf.Variable(tf.constant(0.0, shape=[featureNum]), name='{}_b'.format(name))
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
    layer_activations.append((tf.get_collection('conv1_1')[0], 224*224*64))
    layer_activations.append((tf.get_collection('conv1_2')[0], 224*224*64))
    layer_activations.append((tf.get_collection('conv2_1')[0], 112*112*128))
    layer_activations.append((tf.get_collection('conv2_2')[0], 112*112*128))
    layer_activations.append((tf.get_collection('conv3_1')[0], 56*56*256))
    layer_activations.append((tf.get_collection('conv3_2')[0], 56*56*256))
    layer_activations.append((tf.get_collection('conv3_3')[0], 56*56*256))
    layer_activations.append((tf.get_collection('conv3_4')[0], 56*56*256))
    layer_activations.append((tf.get_collection('conv4_1')[0], 28*28*512))
    layer_activations.append((tf.get_collection('conv4_2')[0], 28*28*512))
    layer_activations.append((tf.get_collection('conv4_3')[0], 28*28*512))
    layer_activations.append((tf.get_collection('conv4_4')[0], 28*28*512))
    layer_activations.append((tf.get_collection('conv5_1')[0], 14*14*512))
    layer_activations.append((tf.get_collection('conv5_2')[0], 14*14*512))
    layer_activations.append((tf.get_collection('conv5_3')[0], 14*14*512))
    layer_activations.append((tf.get_collection('conv5_4')[0], 14*14*512))
    layer_activations.append((tf.get_collection('fc6')[0], 4096))
    layer_activations.append((tf.get_collection('fc7')[0], 4096))
    layer_activations.append((tf.get_collection('fc8_w')[0], output_size))
    layer_activations.append((tf.get_collection('fc8_b')[0], output_size))

    feed_dicts = create_feed_dicts(keep_prob, temperature)

    return inputs, outputs, layer_activations, feed_dicts


def load_and_freeze_model(sess, inputs, model_meta, model_checkpoint, batch_size, output_size):
    new_saver = tf.train.import_meta_graph(model_meta)
    new_saver.restore(sess, model_checkpoint)
    
    with tf.variable_scope('vgg'):
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        temperature = tf.placeholder(tf.float32, name='temperature')

        # MAKE SURE INPUTS ARE 224x224 IMAGES, otherwise the layer size below is wrong!!!
        inputs_reshaped = tf.reshape(inputs, [-1, 224, 224, 1])

        with tf.variable_scope('conv_pool_1'):
            conv1_1_w = tf.constant(sess.run(tf.get_collection('conv1_1_w')[0]), name='conv1_1_w')
            conv1_1_b = tf.constant(sess.run(tf.get_collection('conv1_1_b')[0]), name='conv1_1_b')
            conv1_1   = tf.nn.conv2d(inputs_reshaped, conv1_1_w, strides=[1,1,1,1], padding='SAME') + conv1_1_b
            
            conv1_2_w = tf.constant(sess.run(tf.get_collection('conv1_2_w')[0]), name='conv1_2_w')
            conv1_2_b = tf.constant(sess.run(tf.get_collection('conv1_2_b')[0]), name='conv1_2_b')
            conv1_2   = tf.nn.conv2d(conv1_1, conv1_2_w, strides=[1,1,1,1], padding='SAME') + conv1_2_b

            pool1 = maxPoolLayer(conv1_2, 2,2,2,2, name="pool1")

        with tf.variable_scope('conv_pool_2'):
            conv2_1_w = tf.constant(sess.run(tf.get_collection('conv2_1_w')[0]), name='conv2_1_w')
            conv2_1_b = tf.constant(sess.run(tf.get_collection('conv2_1_b')[0]), name='conv2_1_b')
            conv2_1   = tf.nn.conv2d(pool1, conv2_1_w, strides=[1,1,1,1], padding='SAME') + conv2_1_b
            
            conv2_2_w = tf.constant(sess.run(tf.get_collection('conv2_2_w')[0]), name='conv2_2_w')
            conv2_2_b = tf.constant(sess.run(tf.get_collection('conv2_2_b')[0]), name='conv2_2_b')
            conv2_2   = tf.nn.conv2d(conv2_1, conv2_2_w, strides=[1,1,1,1], padding='SAME') + conv2_2_b
            
            pool2 = maxPoolLayer(conv2_2, 2,2,2,2,  "pool2")

        with tf.variable_scope('conv_pool_3'):
             
            conv3_1_w = tf.constant(sess.run(tf.get_collection('conv3_1_w')[0]), name='conv3_1_w')
            conv3_1_b = tf.constant(sess.run(tf.get_collection('conv3_1_b')[0]), name='conv3_1_b')
            conv3_1   = tf.nn.conv2d(pool2, conv3_1_w, strides=[1,1,1,1], padding='SAME') + conv3_1_b
            
            conv3_2_w = tf.constant(sess.run(tf.get_collection('conv3_2_w')[0]), name='conv3_2_w')
            conv3_2_b = tf.constant(sess.run(tf.get_collection('conv3_2_b')[0]), name='conv3_2_b')
            conv3_2   = tf.nn.conv2d(conv3_1, conv3_2_w, strides=[1,1,1,1], padding='SAME') + conv3_2_b
            
            conv3_3_w = tf.constant(sess.run(tf.get_collection('conv3_3_w')[0]), name='conv3_3_w')
            conv3_3_b = tf.constant(sess.run(tf.get_collection('conv3_3_b')[0]), name='conv3_3_b')
            conv3_3   = tf.nn.conv2d(conv3_2, conv3_3_w, strides=[1,1,1,1], padding='SAME') + conv3_3_b
            
            conv3_4_w = tf.constant(sess.run(tf.get_collection('conv3_4_w')[0]), name='conv3_4_w')
            conv3_4_b = tf.constant(sess.run(tf.get_collection('conv3_4_b')[0]), name='conv3_4_b')
            conv3_4   = tf.nn.conv2d(conv3_3, conv3_4_w, strides=[1,1,1,1], padding='SAME') + conv3_4_b
            
            pool3 = maxPoolLayer(conv3_4, 2, 2, 2, 2, "pool3")

        with tf.variable_scope('conv_pool_4'):
            conv4_1_w = tf.constant(sess.run(tf.get_collection('conv4_1_w')[0]), name='conv4_1_w')
            conv4_1_b = tf.constant(sess.run(tf.get_collection('conv4_1_b')[0]), name='conv4_1_b')
            conv4_1   = tf.nn.conv2d(pool3, conv4_1_w, strides=[1,1,1,1], padding='SAME') + conv4_1_b
            
            conv4_2_w = tf.constant(sess.run(tf.get_collection('conv4_2_w')[0]), name='conv4_2_w')
            conv4_2_b = tf.constant(sess.run(tf.get_collection('conv4_2_b')[0]), name='conv4_2_b')
            conv4_2   = tf.nn.conv2d(conv4_1, conv4_2_w, strides=[1,1,1,1], padding='SAME') + conv4_2_b
            
            conv4_3_w = tf.constant(sess.run(tf.get_collection('conv4_3_w')[0]), name='conv4_3_w')
            conv4_3_b = tf.constant(sess.run(tf.get_collection('conv4_3_b')[0]), name='conv4_3_b')
            conv4_3   = tf.nn.conv2d(conv4_2, conv4_3_w, strides=[1,1,1,1], padding='SAME') + conv4_3_b
            
            conv4_4_w = tf.constant(sess.run(tf.get_collection('conv4_4_w')[0]), name='conv4_4_w')
            conv4_4_b = tf.constant(sess.run(tf.get_collection('conv4_4_b')[0]), name='conv4_4_b')
            conv4_4   = tf.nn.conv2d(conv4_3, conv4_4_w, strides=[1,1,1,1], padding='SAME') + conv4_4_b
            pool4 = maxPoolLayer(conv4_4, 2, 2, 2, 2, "pool4")

        with tf.variable_scope('conv_pool_5'):
            conv5_1_w = tf.constant(sess.run(tf.get_collection('conv5_1_w')[0]), name='conv5_1_w')
            conv5_1_b = tf.constant(sess.run(tf.get_collection('conv5_1_b')[0]), name='conv5_1_b')
            conv5_1   = tf.nn.conv2d(pool4, conv5_1_w, strides=[1,1,1,1], padding='SAME') + conv5_1_b
            
            conv5_2_w = tf.constant(sess.run(tf.get_collection('conv5_2_w')[0]), name='conv5_2_w')
            conv5_2_b = tf.constant(sess.run(tf.get_collection('conv5_2_b')[0]), name='conv5_2_b')
            conv5_2   = tf.nn.conv2d(conv5_1, conv5_2_w, strides=[1,1,1,1], padding='SAME') + conv5_2_b
            
            conv5_3_w = tf.constant(sess.run(tf.get_collection('conv5_3_w')[0]), name='conv5_3_w')
            conv5_3_b = tf.constant(sess.run(tf.get_collection('conv5_3_b')[0]), name='conv5_3_b')
            conv5_3   = tf.nn.conv2d(conv5_2, conv5_3_w, strides=[1,1,1,1], padding='SAME') + conv5_3_b

            conv5_4_w = tf.constant(sess.run(tf.get_collection('conv5_4_w')[0]), name='conv5_4_w')
            conv5_4_b = tf.constant(sess.run(tf.get_collection('conv5_4_b')[0]), name='conv5_4_b')
            conv5_4   = tf.nn.conv2d(conv5_3, conv5_4_w, strides=[1,1,1,1], padding='SAME') + conv5_4_b
            pool5 = maxPoolLayer(conv5_4, 2, 2, 2, 2, "pool5")

        with tf.variable_scope('fc_6'):
            fcIn = tf.reshape(pool5, [-1, 7*7*512])
            fc6_w = tf.constant(sess.run(tf.get_collection('fc6_w')[0]), name='fc6_w')
            fc6_b = tf.constant(sess.run(tf.get_collection('fc6_b')[0]), name='fc6_b')
            fc6 = tf.matmul(fcIn, fc6_w) + fc6_b
            
            dropout1 = dropout(fc6, keep_prob)

        with tf.variable_scope('fc_7'):

            fc7_w = tf.constant(sess.run(tf.get_collection('fc7_w')[0]), name='fc7_w')
            fc7_b = tf.constant(sess.run(tf.get_collection('fc7_b')[0]), name='fc7_b')
            fc7 = tf.matmul(dropout1, fc7_w) + fc7_b
            
            dropout2 = dropout(fc7, keep_prob)

        with tf.variable_scope('fc_8'):

            fc8_w = tf.constant(sess.run(tf.get_collection('fc8_w')[0]), name='fc8_w')
            fc8_b = tf.constant(sess.run(tf.get_collection('fc8_b')[0]), name='fc8_b')
            logits = tf.matmul(dropout2, fc8_w) + fc8_b
            logits = tf.div(logits, temperature)
    
    layer_activations = []
    layer_activations.append((tf.get_collection('conv1_1')[0], 224*224*64))
    layer_activations.append((tf.get_collection('conv1_2')[0], 224*224*64))
    layer_activations.append((tf.get_collection('conv2_1')[0], 112*112*128))
    layer_activations.append((tf.get_collection('conv2_2')[0], 112*112*128))
    layer_activations.append((tf.get_collection('conv3_1')[0], 56*56*256))
    layer_activations.append((tf.get_collection('conv3_2')[0], 56*56*256))
    layer_activations.append((tf.get_collection('conv3_3')[0], 56*56*256))
    layer_activations.append((tf.get_collection('conv3_4')[0], 56*56*256))
    layer_activations.append((tf.get_collection('conv4_1')[0], 28*28*512))
    layer_activations.append((tf.get_collection('conv4_2')[0], 28*28*512))
    layer_activations.append((tf.get_collection('conv4_3')[0], 28*28*512))
    layer_activations.append((tf.get_collection('conv4_4')[0], 28*28*512))
    layer_activations.append((tf.get_collection('conv5_1')[0], 14*14*512))
    layer_activations.append((tf.get_collection('conv5_2')[0], 14*14*512))
    layer_activations.append((tf.get_collection('conv5_3')[0], 14*14*512))
    layer_activations.append((tf.get_collection('conv5_4')[0], 14*14*512))
    layer_activations.append((tf.get_collection('fc6')[0], 4096))
    layer_activations.append((tf.get_collection('fc7')[0], 4096))
    layer_activations.append((tf.get_collection('fc8_w')[0], output_size))
    layer_activations.append((tf.get_collection('fc8_b')[0], output_size))
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
