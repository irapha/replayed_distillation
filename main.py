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

    # create graph
    input_size, output_size = d.get_io_size(FLAGS.dataset)
    #  keep_inp, keep, temp, labels_temp = u.create_optional_params()
    temp = tf.placeholder(tf.float32, name='temp')

    inp, labels, keep_inp, keep, labels_temp = p.get(FLAGS.procedure).create_placeholders(input_size, output_size, None)
    labels_evaldistill = tf.placeholder(tf.float32, [None, output_size], name='labels_evaldistill')

    out = m.get(FLAGS.model).create_model(inp, output_size, keep_inp, keep, temp)

    #  tf.add_to_collection('input', inp)
    #  tf.add_to_collection('output', out)
    #  tf.add_to_collection('keep', keep)
    #  tf.add_to_collection('keep_inp', keep_inp)
    #  tf.add_to_collection('labels_temp', temp) # not wrong dw

    loss, train_step = u.create_train_ops(out, labels)
    loss_evaldistill, _ = u.create_train_ops(out, labels_evaldistill, scope='evaldistill')

    accuracy, top5 = u.create_eval_ops(out, labels)
    accuracy_evaldistill, top5_evaldistill = u.create_eval_ops(out, labels_evaldistill, scope='evaldistill')
    summary_op = u.create_summary_ops(loss, accuracy, top5)
    summary_op_evaldistill = u.create_summary_ops(loss_evaldistill, accuracy_evaldistill, top5_evaldistill)

    # initialize dataset interface
    data = d.get(FLAGS.dataset)

    # initialize session
    sess = tf.Session(u.get_sess_config(use_gpu=True))
    sess.run(tf.global_variables_initializer())

    # run training procedure
    p.get(FLAGS.procedure).run(sess, FLAGS, data,
            (inp, labels, keep_inp, keep, temp, labels_temp, labels_evaldistill), train_step, summary_op, summary_op_evaldistill)

    # save log
    u.save_log(log, FLAGS.summary_folder, FLAGS.run_name, FLAGS.log_file)

