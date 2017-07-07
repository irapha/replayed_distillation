# python main.py --run_name=hinton800_mnist --dataset=mnist --model=hinton800 --procedure=train

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

flags.DEFINE_integer('epochs', 10, 'Number of training epochs')
flags.DEFINE_integer('train_batch_size', 64, 'number of examples to be used for training')
flags.DEFINE_integer('test_batch_size', 64, 'number of examples to be used for testing')
flags.DEFINE_integer('eval_interval', 100, 'Number of training steps between test set evaluations')
flags.DEFINE_integer('checkpoint_interval', 1000, 'Number of steps between checkpoints')


if __name__ == '__main__':
    # initial bookkeeping
    log = u.get_logger(FLAGS)
    np.random.seed(FLAGS.rng_seed)
    tf.set_random_seed(FLAGS.rng_seed)

    # initialize session
    sess = tf.Session(u.get_sess_config(use_gpu=True))

    # initialize dataset interface
    data = d.get(FLAGS.dataset)

    # run procedure (this will create and train graphs, etc).
    p.get(FLAGS.procedure).run(sess, FLAGS, data)

    # save log
    u.save_log(log, FLAGS.summary_folder, FLAGS.run_name, FLAGS.log_file)

