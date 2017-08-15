"""
This training procedure will train a network from scratch using raw data from
the dataset, and save the checkpoints.
"""
import numpy as np
import os
import tensorflow as tf
import models as m
import utils as u


def run(sess, f, data):
    # create graph
    input_size, output_size = data.io_shape
    inputs = tf.placeholder(tf.float32, [None, input_size], name='inputs')
    outputs, _, feed_dicts = m.get(f.model).create_model(inputs, output_size)

    labels = tf.placeholder(tf.float32, [None, output_size], name='labels')
    loss, train_step = create_train_ops(outputs, labels, loss=f.loss)
    accuracy = create_eval_ops(outputs, labels, loss=f.loss)
    summary_op = create_summary_ops(loss, accuracy)

    # only initialize non-initialized vars:
    u.init_uninitted_vars(sess)
    # (this is not super important for training, but its very important
    # in optimize, and in distill)

    saver = tf.train.Saver(tf.global_variables())

    summary_dir = os.path.join(f.summary_folder, f.run_name, 'train')
    train_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'train'), sess.graph)
    trainbatch_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'train_batch'), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'test'), sess.graph)

    with sess.as_default():
        global_step = 0

        for i in range(f.epochs):
            print('Epoch: {}'.format(i))
            for batch_x, batch_y in data.train_epoch_in_batches(f.train_batch_size):
                summary, _ = sess.run([summary_op, train_step],
                        feed_dict={**feed_dicts['train'],
                                   inputs: batch_x, labels: batch_y})
                trainbatch_writer.add_summary(summary, global_step)

                if global_step % f.eval_interval == 0:
                    # eval test set
                    summaries = []
                    for test_batch_x, test_batch_y in data.test_epoch_in_batches(f.test_batch_size):
                        summary = sess.run(summary_op,
                                feed_dict={**feed_dicts['eval'],
                                           inputs: test_batch_x, labels: test_batch_y})
                        summaries.append(summary)
                    test_writer.add_summary(u.merge_summary_list(summaries, True), global_step)

                    # eval train set
                    summaries = []
                    for train_batch_x, train_batch_y in data.train_epoch_in_batches(f.train_batch_size):
                        summary = sess.run(summary_op,
                                feed_dict={**feed_dicts['eval'],
                                           inputs: train_batch_x, labels: train_batch_y})
                        summaries.append(summary)
                    train_writer.add_summary(u.merge_summary_list(summaries, True), global_step)

                global_step += 1

                if global_step % f.checkpoint_interval == 0:
                    checkpoint_dir = os.path.join(summary_dir, 'checkpoint/')
                    u.ensure_dir_exists(checkpoint_dir)
                    checkpoint_file = os.path.join(checkpoint_dir, f.model)
                    saved_file = saver.save(sess, checkpoint_file, global_step=global_step)

    print('saved model at {}'.format(saved_file))

def create_train_ops(h, labels, scope='train_ops', loss='xent'):
    if loss == 'xent':
        with tf.variable_scope('xent_' + scope):
            loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=h, name='sftmax_xent'))
    elif loss == 'attrxent':
        with tf.variable_scope('attrxent_' + scope):
            # first reshape output to have one more dim with 2 attrs
            h = tf.reshape(h, (-1, 2))
            labels = tf.reshape(labels, (-1, 2))
            loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=h, name='sftmax_attrxent'))
    elif loss == 'mse':
        with tf.variable_scope('mse_' + scope):
            loss = tf.losses.mean_squared_error(labels=labels, predictions=tf.nn.relu(h))

    with tf.variable_scope('opt_' + scope):
        train_step = tf.train.AdamOptimizer().minimize(loss)

    return loss, train_step

def create_eval_ops(y, y_, scope='train_ops', loss='xent'):
    with tf.variable_scope('eval_' + scope):
        if loss == 'xent':
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        elif loss == 'attrxent':
            y = tf.reshape(y, (-1, 2))
            y_ = tf.reshape(y_, (-1, 2))
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        elif loss == 'mse':
            correct_prediction = tf.equal(tf.sign(y), y_)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def create_summary_ops(loss, accuracy):
    loss_summary_op = tf.summary.scalar('loss', loss)
    accuracy_summary_op = tf.summary.scalar('accuracy', accuracy)
    return tf.summary.merge([loss_summary_op, accuracy_summary_op])

