
def get(optimization_objective):
    if optimization_objective == 'top_layer':
        return top_layer
    elif optimization_objective == 'all_layers':
        return all_layers
    elif optimization_objective == 'spectral_all_layers':
        return spectral_all_layers
    elif optimization_objective == 'spectral_layer_pairs':
        return spectral_layer_pairs
    else:
        raise NotImplemented('This procedure not implemented yet')

class top_layer:
    def __init__(self, layer_activations):
        tensor, size = layer_activations[-1]
        self.top_layer_size = size
        self.top_layer_placeholder = tf.placeholder(tf.float32, [None, size], name='{}_placeholder'.format(tensor))
        recreate_loss = tf.reduce_mean(tf.pow(tf.nn.relu(self.top_layer_placeholder) - tf.nn.relu(tensor), 2))
        self.recreate_op = tf.train.AdamOptimizer(learning_rate=0.07).minimize(recreate_loss)

    def sample_from_stats(self, stats, clas, batch_size, feed_dict={}):
        top_layer_stats = stats[-1]
        sampled_values = sample_from_stats(top_layer_stats, clas, batch_size, self.top_layer_size)
        feed_dict[self.top_layer_placeholder] = sampled_values
        return feed_dict

class all_layers:
    def __init__(self, layer_activations):
        self.layer_placeholders = []
        self.layer_sizes = []
        recreate_loss = 0.0
        for tensor, size in layer_activations:
            placeholder = tf.placeholder(tf.float32, [None, size], name='{}_placeholder'.format(tensor))
            self.layer_placeholders.append(placeholder)
            self.layer_placeholders.append(size)

            recreate_loss += (1.0 / size) * tf.reduce_mean(tf.pow(tf.nn.relu(placeholder) - tf.nn.relu(tensor), 2))
        self.recreate_op = tf.train.AdamOptimizer(learning_rate=0.07).minimize(recreate_loss)

    def sample_from_stats(self, stats, clas, batch_size, feed_dict={}):
        for stat, placeholder, size in zip(stats, self.layer_placeholders, self.layer_sizes):
            sampled_values = sample_from_stats(stat, clas, batch_size, size)
            feed_dict[placeholder] = sampled_values
        return feed_dict


class spectral_all_layers:
    def __init__(self):
        raise NotImplemented('TODO(sfenu3): implement. also see sample_from_stats()')

class spectral_layer_pairs:
    def __init__(self):
        raise NotImplemented('TODO(sfenu3): implement. also see sample_from_stats()')

def sample_from_stats(stats, clas, batch_size, out_size):
    # TODO(sfenu3): if you modify what stats are being saved in compute_stats
    # procedure, modify the line below too.
    means, cov, _ = stats
    out_size = means[list(means.keys())[0]].shape[0]
    gauss = np.random.normal(size=(batch_size, out_size))
    pre_sftmx = means[clas] + np.matmul(gauss, cov[clas])
    return pre_sftmx
