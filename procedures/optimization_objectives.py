
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
        self.layer_placeholder = tf.placeholder(tf.float32, [None, size], name='{}_placeholder'.format(tensor))
        act_sft =
        recreate_loss = tf.reduce_mean(tf.pow(tf.nn.relu(self.latent_placeholder)- tf.nn.relu(tensor), 2))
        self.recreate_op = tf.train.AdamOptimizer(learning_rate=0.07).minimize(recreate_loss)

    def sample_from_stats(self, stats, feed_dict={}):
        raise NotImplemented('Oops')

class all_layers:
    def __init__(self, layer_activations):
        self.layer_placeholders = []
        recreate_loss = 0.0
        for tensor, size in layer_activations:
            placeholder = tf.placeholder(tf.float32, [None, size], name='{}_placeholder'.format(tensor))
            self.layer_placeholders.append(placeholder)
            recreate_loss += (1.0 / size) * tf.reduce_mean(tf.pow(tf.nn.relu(placeholder) - tf.nn.relu(tensor), 2))
        self.recreate_op = tf.train.AdamOptimizer(learning_rate=0.07).minimize(recreate_loss)


class spectral_all_layers:
    def __init__(self):
        raise NotImplemented('TODO(sfenu3): implement')

class spectral_layer_pairs:
    def __init__(self):
        raise NotImplemented('TODO(sfenu3): implement')
