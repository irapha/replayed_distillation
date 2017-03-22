"""
This training procedure will train a student network from scratch using
replayed data sampled from a teacher model, and the sampled activations from
that teacher.
"""
import numpy as np
import os
import sys
import tensorflow as tf

import models as m
import utils as u

from utils import ensure_dir_exists

MODEL_META = 'summaries/hinton1200_mnist_withcollect/checkpoint/hinton1200-8000.meta'
MODEL_CHECKPOINT = 'summaries/hinton1200_mnist_withcollect/checkpoint/hinton1200-8000'

def merge_summary_list(summary_list, do_print=False):
    summary_dict = {}

    for summary in summary_list:
        summary_proto = tf.Summary()
        summary_proto.ParseFromString(summary)

        for val in summary_proto.value:
            if val.tag not in summary_dict:
                summary_dict[val.tag] = []
            summary_dict[val.tag].append(val.simple_value)

    # get mean of each tag
    for k, v in summary_dict.items():
        summary_dict[k] = np.mean(v)

    if do_print:
        print(summary_dict)

    # create final Summary with mean of values
    final_summary = tf.Summary()
    final_summary.ParseFromString(summary_list[0])

    for i, val in enumerate(final_summary.value):
        final_summary.value[i].simple_value = summary_dict[val.tag]

    return final_summary

def compute_class_statistics(sess, inp, keep_inp, keep, data, temp):
    all_activations = {}
    for batch_x, batch_y in data.train_epoch_in_batches(50):
        #  batch_out = sess.run('labels_sftmx/Reshape_1:0',
        batch_out = sess.run('784-1200-1200-10/temp/div:0',
                feed_dict={inp: batch_x, keep_inp: 1.0, keep: 1.0, 'temp_1:0': temp})

        for act, y in zip(batch_out, batch_y):
            clas = np.where(y == 1)[0][0]
            if clas not in all_activations:
                all_activations[clas] = []
            all_activations[clas].append(act)


    # consolidate them:
    means = {}
    cov = {}
    for k, v in all_activations.items():
        means[k] = np.mean(v, axis=0)
        cov[k] = np.cov(np.transpose(v))

    #  print('0')
    #  print(all_activations[0])
    #  print('min: {}'.format(np.min(all_activations[0], axis=0)))
    #  print('max: {}'.format(np.max(all_activations[0], axis=0)))
    #  print('mean: {}'.format(means[0]))
    #  print('var: {}'.format(np.var(all_activations[0], axis=0)))
    #  print(cov[0])

    return means, cov

def sample_from_stats(stats, clas, batch_size, out_size):
    means, cov = stats
    out_size = means[list(means.keys())[0]].shape[0]
    gauss = np.random.normal(size=(batch_size, out_size))
    return means[clas] + np.matmul(gauss, cov[clas])

def sample_images(sess, stats, clas, batch_size, latent_placeholder, recreate_op):
    latent = sample_from_stats(stats, clas, batch_size, 10)

    recreated_imgs = sess.run(recreate_op,
            feed_dict={latent_placeholder: latent})

    return recreated_imgs


def run(sess, f, data, placeholders, train_step, summary_op, summary_op_evaldistill):
    inp, labels, keep_inp, keep, temp, labels_temp, labels_evaldistill = placeholders

    # step1: create dict of teacher model class statistics (as seen in Neurogenesis Deep Learning)
    stats = compute_class_statistics(sess, inp, keep_inp, keep, data, 8.0)
    latent_placeholder = tf.placeholder(tf.float32, [None, 10], name='latent_placeholder')
    recreate_op = m.get('hinton1200').create_inverse_model(sess, latent_placeholder)
    with tf.variable_scope('stopp'):
        recreate_op = tf.stop_gradient(recreate_op)
    u.init_uninitted_vars(sess)

    saver = tf.train.Saver(tf.global_variables())

    summary_dir = os.path.join(f.summary_folder, f.run_name)
    train_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'train'), sess.graph)
    trainbatch_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'train_batch'), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'test'), sess.graph)

    with sess.as_default():
        global_step = 0


        print(sample_images(sess, stats, 0, 1, latent_placeholder, recreate_op))


        sys.exit(0)


        for i in range(f.epochs):
            print('Epoch: {}'.format(i))
            for batch_x, batch_y in data.train_epoch_in_batches(f.train_batch_size):
                summary, _ = sess.run([summary_op_evaldistill, train_step],
                        feed_dict={inp: batch_x,
                            labels_evaldistill: batch_y,
                            keep_inp: 1.0, keep: 1.0,
                            'temp_1:0': 8.0, temp: 8.0})

                trainbatch_writer.add_summary(summary, global_step)

                if global_step % f.eval_interval == 0:
                    # eval test
                    summaries = []
                    for test_batch_x, test_batch_y in data.test_epoch_in_batches(f.test_batch_size):
                        summary = sess.run(summary_op_evaldistill,
                                feed_dict={inp: test_batch_x,
                                    labels_evaldistill: test_batch_y,
                                    keep_inp: 1.0, keep: 1.0,
                                    'temp_1:0': 1.0, temp: 1.0})
                        summaries.append(summary)
                    test_writer.add_summary(merge_summary_list(summaries, True), global_step)

                    # eval train
                    summaries = []
                    for train_batch_x, train_batch_y in data.train_epoch_in_batches(f.train_batch_size):
                        summary = sess.run(summary_op_evaldistill,
                                feed_dict={inp: train_batch_x,
                                    labels_evaldistill: train_batch_y,
                                    keep_inp: 1.0, keep: 1.0,
                                    'temp_1:0': 1.0, temp: 1.0})
                        summaries.append(summary)
                    train_writer.add_summary(merge_summary_list(summaries, True), global_step)

                global_step += 1

                if global_step % f.checkpoint_interval == 0:
                    checkpoint_dir = os.path.join(summary_dir, 'checkpoint/')
                    ensure_dir_exists(checkpoint_dir)
                    checkpoint_file = os.path.join(checkpoint_dir, f.model)
                    saver.save(sess, checkpoint_file, global_step=global_step)

def create_placeholders(sess, input_size, output_size, _):
    new_saver = tf.train.import_meta_graph(MODEL_META)
    new_saver.restore(sess, MODEL_CHECKPOINT)

    inp = tf.get_collection('input')[0]
    out = tf.get_collection('output')[0]
    keep_inp = tf.get_collection('keep_inp')[0]
    keep = tf.get_collection('keep')[0]
    labels_temp = tf.get_collection('labels_temp')[0]

    with tf.variable_scope('labels_sftmx'):
        labels = tf.nn.softmax(out)

    with tf.variable_scope('stoppp'):
        labels = tf.stop_gradient(labels)

    return inp, labels, keep_inp, keep, labels_temp
