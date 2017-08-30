"""
This training procedure will train a student network from scratch using raw data
from the dataset, and activations from a teacher model.
"""
import numpy as np
import os
import tensorflow as tf
import utils as u
import models as m
import datasets as d

from utils import ensure_dir_exists, merge_summary_list

MODEL_META = 'summaries/hinton1200_mnist_withcollect/checkpoint/hinton1200-8000.meta'
MODEL_CHECKPOINT = 'summaries/hinton1200_mnist_withcollect/checkpoint/hinton1200-8000'


def run(sess, f, data):
    # load data that will be used for evaluating the distillation process
    eval_data = d.get(f.eval_dataset, f)

    # load teacher graph
    _, output_size = data.io_shape
    inputs, teacher_outputs, _, teacher_feed_dicts = m.get(f.model).load_model(sess, f.model_meta, f.model_checkpoint, output_size)
    teacher_outputs = tf.stop_gradient(tf.nn.softmax(teacher_outputs))

    # create student graph
    outputs, _, feed_dicts = m.get(f.model).create_model(inputs, output_size)

    loss, train_step = create_train_ops(outputs, labels, lr=f.lr, loss=f.loss)
    accuracy = create_eval_ops(outputs, teacher_outputs)
    summary_op = create_summary_ops(loss, accuracy)

    # only initialize non-initialized vars:
    u.init_uninitted_vars(sess)
    # (this is very important in distill: we don't want to reinit teacher model)

    saver = tf.train.Saver(tf.global_variables())

    summary_dir = os.path.join(f.summary_folder, f.run_name, 'distill')
    train_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'train'), sess.graph)
    trainbatch_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'train_batch'), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'test'), sess.graph)

    with sess.as_default():
        global_step = 0

        for i in range(f.epochs):
            print('Epoch: {}'.format(i))
            for batch_x, _ in data.train_epoch_in_batches(f.train_batch_size):
                # train step. we don't need to feed batch_y because the student
                # is being trained to mimic the teacher's temperature-scaled
                # activations.
                summary, _ = sess.run([summary_op, train_step],
                        feed_dict={**teacher_feed_dicts['distill'],
                                   **feed_dicts['distill'],
                                   inputs: batch_x})
                trainbatch_writer.add_summary(summary, global_step)

                if global_step % f.eval_interval == 0:
                    # eval test
                    summaries = []
                    for test_batch_x, test_batch_y in eval_data.test_epoch_in_batches(f.test_batch_size):
                        summary = sess.run(summary_op,
                                feed_dict={**teacher_feed_dicts['distill'],
                                           **feed_dicts['distill'],
                                           inputs: test_batch_x})
                        summaries.append(summary)
                    test_writer.add_summary(merge_summary_list(summaries, True), global_step)

                    # eval train
                    summaries = []
                    for train_batch_x, train_batch_y in data.train_epoch_in_batches(f.train_batch_size):
                        summary = sess.run(summary_op,
                                feed_dict={**teacher_feed_dicts['distill'],
                                           **feed_dicts['distill'],
                                           inputs: train_batch_x})
                        summaries.append(summary)
                    train_writer.add_summary(merge_summary_list(summaries, True), global_step)

                global_step += 1

                if global_step % f.checkpoint_interval == 0:
                    checkpoint_dir = os.path.join(summary_dir, 'checkpoint/')
                    ensure_dir_exists(checkpoint_dir)
                    checkpoint_file = os.path.join(checkpoint_dir, f.model)
                    saver.save(sess, checkpoint_file, global_step=global_step)

    print('distilled model saved in {}'.format(checkpoint_file))

def create_train_ops(h, labels, lr=0.001, scope='train_ops', loss='xent'):
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
        train_step = tf.train.AdamOptimizer(learning_rate=float(lr)).minimize(loss)

    return loss, train_step

def create_eval_ops(y, y_, scope='train_ops'):
    with tf.variable_scope('eval_' + scope):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def create_summary_ops(loss, accuracy):
    loss_summary_op = tf.summary.scalar('loss', loss)
    accuracy_summary_op = tf.summary.scalar('accuracy', accuracy)
    return tf.summary.merge([loss_summary_op, accuracy_summary_op])
