# 784-1200-1200-10 model
# with 50% dropout on hidden and 20% on visible nodes
import tensorflow as tf
import numpy as np


def create_model(inputs, output_size):
    layer_activations = []

    with tf.variable_scope('784-1200-1200-10'):
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
        tf.add_to_collection('fc1_w', w)
        tf.add_to_collection('fc1_b', b)
        tf.add_to_collection('fc1', h)

        with tf.variable_scope('fc2'):
            w = tf.Variable(tf.truncated_normal([1200, 1200]), name='w')
            b = tf.Variable(tf.constant(0.1, shape=[1200]), name='b')
            h = tf.matmul(z_drop, w) + b
            z = tf.nn.relu(h, name='relu')
            z_drop = tf.nn.dropout(z, keep_prob=keep_prob)

        layer_activations.append((h, 1200))
        tf.add_to_collection('fc2_w', w)
        tf.add_to_collection('fc2_b', b)
        tf.add_to_collection('fc2', h)

        with tf.variable_scope('fc3'):
            w = tf.Variable(tf.truncated_normal([1200, output_size]), name='w')
            b = tf.Variable(tf.constant(0.1, shape=[output_size]), name='b')
            h = tf.matmul(z_drop, w) + b

        tf.add_to_collection('fc3_w', w)
        tf.add_to_collection('fc3_b', b)

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

    return h_soft, layer_activations, feed_dicts

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

def load_and_freeze_model(sess, inputs, model_meta, model_checkpoint, batch_size, output_size):
    new_saver = tf.train.import_meta_graph(model_meta)
    new_saver.restore(sess, model_checkpoint)
    temp = tf.placeholder(tf.float32, name='temperature')

    layer_activations = []
    dropout_filters = [] # tuples (filter_placeholder, filter_assign_op)
    dropout_rescales = []

    with tf.variable_scope('784-1200-1200-10_const'):
        with tf.variable_scope('inp_drop'):
            dropout_rescale = tf.placeholder(tf.float32, name='dropout_rescale')
            dropout_rescales.append(dropout_rescale)

            dropout_filter_placeholder = tf.placeholder(tf.float32, [batch_size, 784], name='drop_place')
            dropout_filter_var = tf.Variable(tf.zeros([batch_size, 784]), name='drop_var')
            dropout_filter_assign_op = tf.assign(dropout_filter_var, dropout_filter_placeholder)
            # here we manually apply dropout. this way, we can fix the dropout
            # filter in the all_layers_dropout optimization_objective, or set
            # it to an matrix of ones (identity function for element-wise
            # multiply) in the other objectives, using the assign_op above.
            inputs = tf.multiply(inputs, dropout_filter_var)
            # a model trained with dropout, when not using a filter that drops
            # neurons out, needs to rescale the layer's activations by dropout_probability.
            inputs = inputs * dropout_rescale # 1.0 in all_layers_dropout, 0.8 otherwise
            # the above applies to all layers in this model
        dropout_filters.append((dropout_filter_placeholder, dropout_filter_assign_op, (batch_size, 784), 0.8))

        with tf.variable_scope('fc1'):
            dropout_rescale = tf.placeholder(tf.float32, name='dropout_rescale')
            dropout_rescales.append(dropout_rescale)

            dropout_filter_placeholder = tf.placeholder(tf.float32, [batch_size, 1200], name='drop_place')
            dropout_filter_var = tf.Variable(tf.zeros([batch_size, 1200]), name='drop_var')
            dropout_filter_assign_op = tf.assign(dropout_filter_var, dropout_filter_placeholder)
            w = tf.constant(sess.run(tf.get_collection('fc1_w')[0]), name='w')
            b = tf.constant(sess.run(tf.get_collection('fc1_b')[0]), name='b')
            h = tf.matmul(inputs, w) + b
            z = tf.nn.relu(h, name='relu')
            # apply both kinds of dropout (see above)
            z = tf.multiply(z, dropout_filter_var)
            z = z * dropout_rescale # 1.0 in all_layers_dropout, 0.5 otherwise
        dropout_filters.append((dropout_filter_placeholder, dropout_filter_assign_op, (batch_size, 1200), 0.5))
        layer_activations.append((h, 1200))

        with tf.variable_scope('fc2'):
            dropout_rescale = tf.placeholder(tf.float32, name='dropout_rescale')
            dropout_rescales.append(dropout_rescale)

            dropout_filter_placeholder = tf.placeholder(tf.float32, [batch_size, 1200], name='drop_place')
            dropout_filter_var = tf.Variable(tf.zeros([batch_size, 1200]), name='drop_var')
            dropout_filter_assign_op = tf.assign(dropout_filter_var, dropout_filter_placeholder)
            w = tf.constant(sess.run(tf.get_collection('fc2_w')[0]), name='w')
            b = tf.constant(sess.run(tf.get_collection('fc2_b')[0]), name='b')
            h = tf.matmul(z, w) + b
            z = tf.nn.relu(h, name='relu')
            # apply both kinds of dropout (see above)
            z = tf.multiply(z, dropout_filter_var)
            z = z * dropout_rescale # 1.0 in all_layers_dropout, 0.5 otherwise
        dropout_filters.append((dropout_filter_placeholder, dropout_filter_assign_op, (batch_size, 1200), 0.5))
        layer_activations.append((h, 1200))

        with tf.variable_scope('fc3'):
            w = tf.constant(sess.run(tf.get_collection('fc3_w')[0]), name='w')
            b = tf.constant(sess.run(tf.get_collection('fc3_b')[0]), name='b')
            h = tf.div(tf.matmul(z, w) + b, temp)
        layer_activations.append((h, output_size))

    feed_dicts = {}
    feed_dicts['distill'] = {
            temp: 8.0,
            dropout_rescales[0]: 0.8,
            dropout_rescales[1]: 0.5,
            dropout_rescales[2]: 0.5}
    feed_dicts['distill_dropout'] = {
            temp: 8.0,
            dropout_rescales[0]: 1.0,
            dropout_rescales[1]: 1.0,
            dropout_rescales[2]: 1.0}

    return h, layer_activations, feed_dicts, dropout_filters
