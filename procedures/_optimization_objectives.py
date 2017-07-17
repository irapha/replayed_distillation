import tensorflow as tf
import numpy as np


def get(optimization_objective):
    if optimization_objective == 'top_layer':
        return top_layer
    elif optimization_objective == 'all_layers':
        return all_layers
    elif optimization_objective == 'all_layers_dropout':
        return all_layers_dropout
    elif optimization_objective == 'spectral_all_layers':
        return spectral_all_layers
    elif optimization_objective == 'spectral_layer_pairs':
        return spectral_layer_pairs
    else:
        raise NotImplemented('Optimization objective {} not implemented' +
                'yet'.format(optimization_objective))

class top_layer:
    def __init__(self, layer_activations):
        tensor, size = layer_activations[-1]
        self.top_layer_size = size
        self.top_layer_placeholder = tf.placeholder(tf.float32, [None, size],
                name='{}_placeholder'.format(get_name(tensor)))
        recreate_loss = tf.reduce_mean(
                tf.pow(tf.nn.relu(self.top_layer_placeholder) - tf.nn.relu(tensor), 2))
        self.recreate_op = tf.train.AdamOptimizer(learning_rate=0.07).minimize(recreate_loss)

    def sample_from_stats(self, stats, clas, batch_size, feed_dicts=None):
        if not feed_dicts: feed_dict = {}
        else: feed_dict = feed_dicts['distill']
        top_layer_stats = stats[-1]
        sampled_values = sample_from_stats(top_layer_stats, clas, batch_size, self.top_layer_size)
        feed_dict[self.top_layer_placeholder] = sampled_values
        return feed_dict

    def reinitialize_dropout_filters(self, sess, dropout_filters):
        # in this objective, we assign the fixed dropout filters to be arrays
        # of ones (identity). Since the model was trained with dropout, we set
        # the layer-wise rescale factor to be the keep_prob of that layer (this
        # is done by using the 'distill' feed_dict, in sample_from_stats)
        for filter_place, filter_assign_op, shape, _ in dropout_filters:
            sess.run(filter_assign_op, feed_dict={filter_place: _get_dropout_filter(shape, 1.0)})

class all_layers:
    def __init__(self, layer_activations):
        self.layer_placeholders = []
        self.layer_sizes = []
        recreate_loss = 0.0
        for tensor, size in layer_activations:
            placeholder = tf.placeholder(tf.float32, [None, size],
                    name='{}_placeholder'.format(get_name(tensor)))
            self.layer_placeholders.append(placeholder)
            self.layer_sizes.append(size)

            recreate_loss += (1.0 / size) * tf.reduce_mean(
                    tf.pow(tf.nn.relu(placeholder) - tf.nn.relu(tensor), 2))
        self.recreate_op = tf.train.AdamOptimizer(learning_rate=0.07).minimize(recreate_loss)

    def sample_from_stats(self, stats, clas, batch_size, feed_dicts=None):
        if not feed_dicts: feed_dict = {}
        else: feed_dict = feed_dicts['distill']
        for stat, placeholder, size in zip(stats, self.layer_placeholders, self.layer_sizes):
            sampled_values = sample_from_stats(stat, clas, batch_size, size)
            feed_dict[placeholder] = sampled_values
        return feed_dict

    def reinitialize_dropout_filters(self, sess, dropout_filters):
        # in this objective, we assign the fixed dropout filters to be arrays
        # of ones (identity). Since the model was trained with dropout, we set
        # the layer-wise rescale factor to be the keep_prob of that layer (this
        # is done by using the 'distill' feed_dict, in sample_from_stats)
        for filter_place, filter_assign_op, shape, _ in dropout_filters:
            sess.run(filter_assign_op, feed_dict={filter_place: _get_dropout_filter(shape, 1.0)})

class all_layers_dropout:
    def __init__(self, layer_activations):
        self.layer_placeholders = []
        self.layer_sizes = []
        recreate_loss = 0.0
        for tensor, size in layer_activations:
            placeholder = tf.placeholder(tf.float32, [None, size],
                    name='{}_placeholder'.format(get_name(tensor)))
            self.layer_placeholders.append(placeholder)
            self.layer_sizes.append(size)

            recreate_loss += (1.0 / size) * tf.reduce_mean(
                    tf.pow(tf.nn.relu(placeholder) - tf.nn.relu(tensor), 2))
        self.recreate_op = tf.train.AdamOptimizer(learning_rate=0.07).minimize(recreate_loss)

    def sample_from_stats(self, stats, clas, batch_size, feed_dicts=None):
        if not feed_dicts: feed_dict = {}
        else: feed_dict = feed_dicts['distill_dropout']
        for stat, placeholder, size in zip(stats, self.layer_placeholders, self.layer_sizes):
            sampled_values = sample_from_stats(stat, clas, batch_size, size)
            feed_dict[placeholder] = sampled_values
        return feed_dict

    def reinitialize_dropout_filters(self, sess, dropout_filters):
        if len(dropout_filters) == 0:
            raise Exception("dropout_filters can't be empty. Are you using" +
                    "all_layers_dropout with a model that doesn't have" +
                    "dropout?")
        # in this objective, we assign fixed dropout filters that arent arrays
        # of ones. There will be 0s with probability 1/keep_prob. we also use
        # the distill_dropout feed_dict, which means the element-wise rescale
        # factor is 1.0 for all layers
        for filter_place, filter_assign_op, shape, keep_prob in dropout_filters:
            sess.run(filter_assign_op, feed_dict={filter_place: _get_dropout_filter(shape, keep_prob)})

def _get_dropout_filter(shape, keep_prob):
    return np.random.binomial(1, keep_prob, size=shape)


class spectral_all_layers:
    def __init__(self):
        raise NotImplemented('TODO(sfenu3): implement. also see sample_from_stats()')

class spectral_layer_pairs:
    def __init__(self):
        raise NotImplemented('TODO(sfenu3): implement. also see sample_from_stats()')

def sample_from_stats(stats, clas, batch_size, out_size):
    # TODO(sfenu3): if you modify what stats are being saved in compute_stats
    # procedure, modify the line below too.
    means, cov, stddev, shape = stats
    out_size = means[list(means.keys())[0]].shape[0]
    gauss = np.random.normal(size=(batch_size, out_size))
    # Some tensors cant have cov matrices (like conv1 in lenet) because they
    # are not positive definite. For those, we sample with element wise gaussian.
    if cov is None:
        pre_sftmx = means[clas] + np.multiply(gauss, stddev[clas])
    else:
        pre_sftmx = means[clas] + np.matmul(gauss, cov[clas])
    return np.reshape(pre_sftmx, shape)

def get_name(tensor):
    return tensor.name.split(':')[0]
