# 784-800-800-10 model
# with no regularization
import tensorflow as tf

def create_model(inputs, output_size):
    layer_activations = []

    with tf.variable_scope('784-800-800-10'):
        temperature = tf.placeholder(tf.float32, name='temperature')

        with tf.variable_scope('fc1'):
            w = tf.Variable(tf.truncated_normal([int(inputs.get_shape()[-1]), 800]), name='w')
            b = tf.Variable(tf.constant(0.1, shape=[800]), name='b')
            h = tf.matmul(inputs, w) + b
            z = tf.nn.relu(h, name='relu')

        layer_activations.append((h, 800))
        tf.add_to_collection('fc1_w', w)
        tf.add_to_collection('fc1_b', b)
        tf.add_to_collection('fc1', h)

        with tf.variable_scope('fc2'):
            w = tf.Variable(tf.truncated_normal([800, 800]), name='w')
            b = tf.Variable(tf.constant(0.1, shape=[800]), name='b')
            h = tf.matmul(z, w) + b
            z = tf.nn.relu(h, name='relu')

        layer_activations.append((h, 800))
        tf.add_to_collection('fc2_w', w)
        tf.add_to_collection('fc2_b', b)
        tf.add_to_collection('fc2', h)

        with tf.variable_scope('fc3'):
            w = tf.Variable(tf.truncated_normal([800, output_size]), name='w')
            b = tf.Variable(tf.constant(0.1, shape=[output_size]), name='b')
            h = tf.matmul(z, w) + b

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
    tf.add_to_collection('temperature', temperature)

    feed_dicts = create_feed_dicts(temperature)

    return h_soft, layer_activations, feed_dicts

def create_feed_dicts(temperature):
    feed_dicts = {key: {} for key in ['train', 'eval', 'distill']}

    feed_dicts['train'][temperature] = 1.0
    feed_dicts['eval'][temperature] = 1.0
    feed_dicts['distill'][temperature] = 8.0

    return feed_dicts

def load_model(sess, model_meta, model_checkpoint, output_size):
    new_saver = tf.train.import_meta_graph(model_meta)
    new_saver.restore(sess, model_checkpoint)

    inputs = tf.get_collection('inputs')[0]
    outputs = tf.get_collection('outputs')[0]
    temperature = tf.get_collection('temperature')[0]

    layer_activations = []
    layer_activations.append((tf.get_collection('fc1')[0], 800))
    layer_activations.append((tf.get_collection('fc2')[0], 800))
    #  layer_activations.append((outputs, int(outputs.get_shape()[-1])))
    # the above doesn't work because tensorflow 1.0 has a bug where restored
    # variables have get_shape == <unknown>. So we just take the output_size
    # from dataset. It's messier but it works.
    layer_activations.append((outputs, output_size))

    feed_dicts = create_feed_dicts(temperature)

    return inputs, outputs, layer_activations, feed_dicts
