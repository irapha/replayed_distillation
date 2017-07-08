# 784-1200-1200-10 model
# with 50% dropout on hidden and 20% on visible nodes
import tensorflow as tf
import numpy as np


def create_model(input_size, output_size):
    layer_activations = []

    with tf.variable_scope('784-1200-1200-10'):
        inputs = tf.placeholder(tf.float32, [None, input_size], name='inputs')

        keep_prob_input = tf.placeholder(tf.float32, name='keep_prob_input')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        temperature = tf.placeholder(tf.float32, name='temperature')

        with tf.variable_scope('inp_drop'):
            inp_drop = tf.nn.dropout(inputs, keep_prob=keep_prob_input)

        with tf.variable_scope('fc1'):
            w = tf.Variable(tf.truncated_normal([int(inputs.get_shape()[-1]), 1200]), name='w')
            b = tf.Variable(tf.constant(0.1, shape=[1200]), name='b')
            h = tf.matmul(inp_drop, w) + b
            z = tf.nn.relu(h, name='relu')
            z_drop = tf.nn.dropout(z, keep_prob=keep_prob)
        layer_activations.append((h, 1200))
        tf.add_to_collection('fc1', inputs)

        with tf.variable_scope('fc2'):
            w = tf.Variable(tf.truncated_normal([1200, 1200]), name='w')
            b = tf.Variable(tf.constant(0.1, shape=[1200]), name='b')
            h = tf.matmul(z_drop, w) + b
            z = tf.nn.relu(h, name='relu')
            z_drop = tf.nn.dropout(z, keep_prob=keep_prob)
        layer_activations.append((h, 1200))
        tf.add_to_collection('fc2', inputs)

        with tf.variable_scope('fc3'):
            w = tf.Variable(tf.truncated_normal([1200, output_size]), name='w')
            b = tf.Variable(tf.constant(0.1, shape=[output_size]), name='b')
            h = tf.matmul(z_drop, w) + b

        with tf.variable_scope('temp'):
            h_soft = tf.div(h, temperature)
        layer_activations.append((h_soft, output_size))

    # if procedure is train, then this model will be the teacher.
    # stats and optimize both load up saved models
    # if procedure is distill, then we already loaded up a saved model (teacher)
    # and we're creating this graph to be the student.
    tf.add_to_collection('inputs', inputs)
    tf.add_to_collection('outputs', h_soft)
    tf.add_to_collection('keep_prob_input', keep_prob_input)
    tf.add_to_collection('keep_prob', keep_prob)
    tf.add_to_collection('temperature', temperature)

    feed_dicts = create_feed_dicts(keep_prob_input, keep_prob, temperature)

    return inputs, h_soft, layer_activations, feed_dicts

def create_feed_dicts(keep_prob_input, keep_prob, temperature):
    feed_dicts = {key: {} for key in ['train', 'eval', 'distill']}

    feed_dicts['train'][keep_prob_input] = 0.8
    feed_dicts['train'][keep_prob] = 0.5
    feed_dicts['train'][temperature] = 1.0

    feed_dicts['eval'][keep_prob_input] = 1.0
    feed_dicts['eval'][keep_prob] = 1.0
    feed_dicts['eval'][temperature] = 1.0

    feed_dicts['distill'][keep_prob_input] = 1.0
    feed_dicts['distill'][keep_prob] = 1.0
    feed_dicts['distill'][temperature] = 8.0

    return feed_dicts

def load_model(sess, model_meta, model_checkpoint, output_size):
    new_saver = tf.train.import_meta_graph(model_meta)
    new_saver.restore(sess, model_checkpoint)

    inputs = tf.get_collection('inputs')[0]
    outputs = tf.get_collection('outputs')[0]
    keep_prob_input = tf.get_collection('keep_prob_input')[0]
    keep_prob = tf.get_collection('keep_prob')[0]
    temperature = tf.get_collection('temperature')[0]

    layer_activations = []
    layer_activations.append((tf.get_collection('fc1')[0], 1200))
    layer_activations.append((tf.get_collection('fc2')[0], 1200))
    #  layer_activations.append((outputs, int(outputs.get_shape()[-1])))
    # the above doesn't work because tensorflow 1.0 has a bug where restored
    # variables have get_shape == <unknown>. So we just take the output_size
    # from dataset. It's messier but it works.
    layer_activations.append((outputs, output_size))

    feed_dicts = create_feed_dicts(keep_prob_input, keep_prob, temperature)

    return inputs, outputs, layer_activations, feed_dicts

def create_constant_model(sess, inp):
    # this should both load and create a constant model, and its feed dicts, which will have to be merged and shit later on

    with tf.variable_scope('784-1200-1200-10_const'):
        with tf.variable_scope('inp_drop'):
            # mere rescaling
            inp = inp * 0.8 # are we using 0.8 anywhere in training?!?!?! (WE were, idk if we stil are in latest exps)

        with tf.variable_scope('fc1'):
            w = tf.constant(sess.run('784-1200-1200-10/fc1/w:0'), name='w')
            b = tf.constant(sess.run('784-1200-1200-10/fc1/b:0'), name='b')
            z = tf.nn.relu(tf.matmul(inp, w) + b, name='relu')
            # mere rescaling
            z = z * 0.5

        with tf.variable_scope('fc2'):
            w = tf.constant(sess.run('784-1200-1200-10/fc2/w:0'), name='w')
            b = tf.constant(sess.run('784-1200-1200-10/fc2/b:0'), name='b')
            z = tf.nn.relu(tf.matmul(z, w) + b, name='relu')
            z = z * 0.5

        with tf.variable_scope('fc3'):
            w = tf.constant(sess.run('784-1200-1200-10/fc3/w:0'), name='w')
            b = tf.constant(sess.run('784-1200-1200-10/fc3/b:0'), name='b')
            h = tf.matmul(z, w) + b

    return h

