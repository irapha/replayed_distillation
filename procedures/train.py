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
    input_size, output_size = data.io_size
    inputs, outputs, feed_dicts = m.get(f.model).create_model(input_size, output_size)

    with tf.variable_scope('train_procedure_ops'):
        labels = tf.placeholder(tf.float32, [None, output_size], name='labels')
        loss, train_step = u.create_train_ops(outputs, labels)
        accuracy = u.create_eval_ops(outputs, labels)
        summary_op = u.create_summary_ops(loss, accuracy)

    # only initialize non-initialized vars:
    u.init_uninitted_vars(sess)
    # (this is not super important for training, but its very important
    # in optimize, and in distill)

    saver = tf.train.Saver(tf.global_variables())

    summary_dir = os.path.join(f.summary_folder, f.run_name)
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
                            #  #  keep_inp: 0.8, keep: 0.5})
                            #  keep_inp: 1.0, keep: 0.5,
                            #  temp: 1.0, labels_temp: 1.0})

                trainbatch_writer.add_summary(summary, global_step)

                if global_step % f.eval_interval == 0:
                    # eval test
                    summaries = []
                    for test_batch_x, test_batch_y in data.test_epoch_in_batches(f.test_batch_size):
                        summary = sess.run(summary_op,
                                feed_dict={**feed_dicts['eval'],
                                           inputs: test_batch_x, labels: test_batch_y})
                                    #  keep_inp: 1.0, keep: 1.0,
                                    #  temp: 1.0, labels_temp: 1.0})
                        summaries.append(summary)
                    test_writer.add_summary(u.merge_summary_list(summaries, True), global_step)

                    # eval train
                    summaries = []
                    for train_batch_x, train_batch_y in data.train_epoch_in_batches(f.train_batch_size):
                        summary = sess.run(summary_op,
                                feed_dict={**feed_dicts['eval'],
                                           inputs: train_batch_x, labels: train_batch_y})
                                    #  keep_inp: 1.0, keep: 1.0, temp: 1.0})
                        summaries.append(summary)
                    train_writer.add_summary(u.merge_summary_list(summaries, True), global_step)

                global_step += 1

                if global_step % f.checkpoint_interval == 0:
                    checkpoint_dir = os.path.join(summary_dir, 'checkpoint/')
                    u.ensure_dir_exists(checkpoint_dir)
                    checkpoint_file = os.path.join(checkpoint_dir, f.model)
                    saver.save(sess, checkpoint_file, global_step=global_step)

