import numpy as np
import tensorflow as tf
import datetime as dt

import models as m
import datasets as d
import utils as u
import procedures as p

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('run_name', '', 'The name of the experimental run')
flags.DEFINE_string('summary_folder', 'summaries/', 'Folder to save summaries and log')
flags.DEFINE_string('log_file', 'log.json', 'Default filename for logs saving')

flags.DEFINE_string('commit', '', 'Commit hash for current experiment')
flags.DEFINE_string('dataset', '', 'ImageNet or MNIST')
flags.DEFINE_string('model', '', 'hinton1200, hinton800, lenet300100, lenet5, alexnet, vgg16')
flags.DEFINE_integer('rng_seed', 42, 'RNG seed, fixed for consistency')
flags.DEFINE_string('procedure', '', 'train, kd, irkd')

flags.DEFINE_integer('epochs', 1, 'Number of training epochs')
flags.DEFINE_integer('train_batch_size', 64, 'number of examples to be used for training')
flags.DEFINE_integer('test_batch_size', 64, 'number of examples to be used for testing')
flags.DEFINE_integer('eval_interval', 100, 'Number of training steps between test set evaluations')
flags.DEFINE_integer('checkpoint_steps', 1000, 'Number of steps between checkpoints')


if __name__ == '__main__':
    # initial bookkeeping
    log = u.get_logger(FLAGS)
    np.random.seed(FLAGS.rng_seed)
    tf.set_random_seed(FLAGS.rng_seed)

    # create graph
    input_size, output_size = d.get_io_size(FLAGS.dataset)
    inp, labels = u.create_placeholders(input_size, output_size)

    keep_inp, keep = u.create_keep_probs()
    out = m.get(FLAGS.model).create_model(inp, output_size, keep_inp, keep)

    train_step = u.create_train_ops(out, labels)
    accuracy = u.create_eval_ops(out, labels)

    # initialize dataset interface
    data = d.get(FLAGS.dataset)

    # initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # run training procedure
    p.get(FLAGS.procedure).run(FLAGS)

    # save log
    u.save_log(log, FLAGS.summary_folder, FLAGS.run_name, FLAGS.log_file)

