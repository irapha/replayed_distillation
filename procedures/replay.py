"""
This training procedure will train a student network from scratch using
replayed data sampled from a teacher model, and the sampled activations from
that teacher.
"""
import numpy as np
import os
import sys
import tensorflow as tf
import cv2
import scipy.stats as st
import scipy

import models as m
import utils as u

from utils import ensure_dir_exists

MODEL_META = 'summaries/hinton1200_mnist_withcollect/checkpoint/hinton1200-8000.meta'
MODEL_CHECKPOINT = 'summaries/hinton1200_mnist_withcollect/checkpoint/hinton1200-8000'

# TODOS:
# - [done] see how relu + l2 reconstruction look like
# - try pruning og model (with og weights) but using optimized examples.
# - [done] try regenerating data without median calculation. Should be quick.
# - [done] use cov matrix again. needs to be stddev tho
# - maybe keep stats for middle of model. then feedforward to get activations
#   for distilling, and optimize input on middle sample.
# - keep stats for all layers actually, then get MSE of each of those, and
#   reconstruct input on all of them.

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

def compute_class_statistics(sess, act_tensor, inp, keep_inp, keep, data, temp, temp_val, stddev=False):
    all_activations = {}
    for batch_x, batch_y in data.train_epoch_in_batches(50):
        #  batch_out = sess.run('labels_sftmx/Reshape_1:0',
        batch_out = sess.run(act_tensor,
                feed_dict={inp: batch_x, keep_inp: 1.0, keep: 1.0, temp: temp_val})

        for act, y in zip(batch_out, batch_y):
            clas = np.where(y == 1)[0][0]
            if clas not in all_activations:
                all_activations[clas] = []
            all_activations[clas].append(act)

    # consolidate them:
    means = {}
    cov = {}
    stdev = {}

    for k, v in all_activations.items():
        means[k] = np.mean(v, axis=0)
        cov[k] = np.linalg.cholesky(np.cov(np.transpose(v)))
        stdev[k] = np.sqrt(np.var(v, axis=0))

    if stddev:
        return means, cov, stdev
    return means, cov

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def sample_from_stats(stats, clas, batch_size, out_size):
    means, cov = stats
    out_size = means[list(means.keys())[0]].shape[0]
    gauss = np.random.normal(size=(batch_size, out_size))
    pre_sftmx = means[clas] + np.matmul(gauss, cov[clas])
    #  pre_sftmx = means[clas] + np.multiply(gauss, cov[clas])
    return pre_sftmx
    #  return [softmax(x) for x in pre_sftmx]

def gkern(size=28, sig=4, noise=0.1):
    g = cv2.getGaussianKernel(size, sig)
    gi = cv2.getGaussianKernel(size, sig).T
    normd = np.matmul(g, gi)
    normd = (1.0 - noise) * (normd / normd.max())
    normd += noise * np.random.uniform(size=[size, size])
    return normd

METHOD = ['onehot', 'onesample', 'manysample'][2]

def sample_images(sess, stats, clas, batch_size, input_placeholder,
        latent_placeholder, input_var, assign_op, recreate_op, data,
        latent_recreated, recreate_loss, reinit_op, temp_recreated, temp_rec_val):

    act_stats,fc2_stats, fc1_stats = stats
    act_placeholder, fc2_placeholder, fc1_placeholder = latent_placeholder

    # these hyper params can be tuned
    num_examples_per_median = 64
    noise = 0.1

    all_medians = []
    all_latents = []

    for i in range(1):
        if METHOD == 'onehot':
            latent = np.zeros([10])
            latent[clas] = 1.0
            all_latents.append(latent)
            latent = [latent] * (num_examples_per_median)
        elif METHOD == 'onesample':
            latent = sample_from_stats(stats, clas, 1, 10)
            all_latents.append(latent[0])
            latent = [latent[0]] * num_examples_per_median
        elif METHOD == 'manysample':
            act_latent = sample_from_stats(act_stats, clas, num_examples_per_median, 10)
            fc2_latent = sample_from_stats(fc2_stats, clas, num_examples_per_median, 1200)
            fc1_latent = sample_from_stats(fc1_stats, clas, num_examples_per_median, 1200)
        print('\tmedian: {}'.format(i))

        # currently setting noise to 0.1 and samples to 64, just bc that will make things easier rn
        sess.run(reinit_op)
        #  input_kernels = [np.reshape(gkern(noise=noise), [784]) for _ in range(num_examples_per_median)]

        # TODO: remove
        #  input_kernels = [np.reshape(np.random.uniform(low=0.0, high=0.2, size=[28, 28]), [784]) for _ in range(num_examples_per_median)]
        input_kernels = [np.reshape(np.random.normal(0.5, 0.1, size=[28, 28]), [784]) for _ in range(num_examples_per_median)]

        sess.run(assign_op, feed_dict={input_placeholder: input_kernels})
        for _ in range(10000):
            _, los = sess.run([recreate_op, recreate_loss],
                    feed_dict={act_placeholder: act_latent,
                        fc2_placeholder: fc2_latent,
                        fc1_placeholder: fc1_latent,
                        temp_recreated: temp_rec_val})

        all_latents.extend(sess.run('const/div:0', feed_dict={temp_recreated: temp_rec_val}))
        all_medians = sess.run(input_var)

    return all_medians, all_latents

def unblockshaped(arr, h, w):
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))

def reshape_to_row(arr):
    grid = np.array([np.reshape(img, (28, 28)) for img in arr])
    return unblockshaped(grid, int(28), int(28 * grid.shape[0]))

def reshape_to_grid(arr):
    grid = np.array([np.reshape(img, (28, 28)) for img in arr])
    size = int(28 * np.sqrt(grid.shape[0]))
    return unblockshaped(grid, size, size)

def compute_optimized_examples(sess, stats, train_batch_size,
        input_placeholder, latent_placeholder, input_var, assign_op,
        recreate_op, data, latent_recreated, recreate_loss, reinit_op,
        temp_recreated, temp_rec_val):
    opt = {}
    for clas in range(10):
        print('clas: {}'.format(clas))
        if clas not in opt:
            opt[clas] = []
        for i in range(10):
            opt[clas].append(sample_images(sess, stats, clas, train_batch_size,
                    input_placeholder, latent_placeholder, input_var, assign_op,
                    recreate_op, data, latent_recreated, recreate_loss, reinit_op,
                    temp_recreated, temp_rec_val))
    return opt


def run(sess, f, data, placeholders, train_step, summary_op, summary_op_evaldistill):
    inp, labels, keep_inp, keep, temp, labels_temp, labels_evaldistill = placeholders

    # create same model with constants instead of vars. And a Variable as input
    with tf.variable_scope('const'):
        act_placeholder = tf.placeholder(tf.float32, [None, 10], name='act_placeholder')
        fc2_placeholder = tf.placeholder(tf.float32, [None, 1200], name='fc2_placeholder')
        fc1_placeholder = tf.placeholder(tf.float32, [None, 1200], name='fc1_placeholder')
        input_placeholder = tf.placeholder(tf.float32, [None, 784], name='input_placeholder')
        input_var = tf.Variable(tf.zeros([f.train_batch_size, 784]), name='recreated_imgs')
        temp_recreated = tf.placeholder(tf.float32, name='temp_recreated')
        sess.run(tf.variables_initializer([input_var], name='init_input'))
        assign_op = tf.assign(input_var, input_placeholder)
        latent_recreated = tf.div(m.get('hinton1200').create_constant_model(sess, input_var), temp_recreated)
        with tf.variable_scope('xent_recreated'):
            if METHOD == 'onehot':
                sft = latent_placeholder
            else:
                #  sft = tf.nn.sigmoid(latent_placeholder)
                #  sft = tf.nn.softmax(latent_placeholder)
                act_sft = tf.nn.relu(act_placeholder)
                fc2_sft = tf.nn.relu(fc2_placeholder)
                fc1_sft = tf.nn.relu(fc1_placeholder)
                #  sft = latent_placeholder
            recreate_loss = (
                    tf.reduce_mean(
                        tf.pow((act_sft - tf.nn.relu(
                            tf.get_default_graph().get_tensor_by_name('784-1200-1200-10/temp/div:0')
                            )), 2)
                        ) +
                    (0.5 *
                        tf.reduce_mean(
                            tf.pow((fc2_sft - tf.nn.relu(
                                tf.get_default_graph().get_tensor_by_name('const/784-1200-1200-10_const/fc2/add:0')
                                )), 2)
                            )) +
                    (0.5 *
                        tf.reduce_mean(
                            tf.pow((fc1_sft - tf.nn.relu(
                                tf.get_default_graph().get_tensor_by_name('const/784-1200-1200-10_const/fc1/add:0')
                                )), 2)
                            ))
                    )
        with tf.variable_scope('opt_recreated'):
            recreate_op = tf.train.AdamOptimizer(learning_rate=0.07).minimize(recreate_loss)

        reinit_op = tf.variables_initializer(u.get_uninitted_vars(sess), name='reinit_op')
        sess.run(reinit_op)

    saver = tf.train.Saver(tf.global_variables())

    summary_dir = os.path.join(f.summary_folder, f.run_name)
    train_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'train'), sess.graph)
    trainbatch_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'train_batch'), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'test'), sess.graph)

    with sess.as_default():
        global_step = 0

        # step1: create dict of teacher model class statistics (as seen in Neurogenesis Deep Learning)
        # TODO: maybe this wrong
        temp_value = 8.0
        # TODO: midlayer
        act_stats = compute_class_statistics(sess, '784-1200-1200-10/temp/div:0', inp, keep_inp, keep, data, 'temp_1:0', temp_value)
        fc2_stats = compute_class_statistics(sess, '784-1200-1200-10/fc2/add:0', inp, keep_inp, keep, data, 'temp_1:0', temp_value)
        fc1_stats = compute_class_statistics(sess, '784-1200-1200-10/fc1/add:0', inp, keep_inp, keep, data, 'temp_1:0', temp_value)
        load_procedure = ['load', 'reconstruct_before', 'reconstruct_fly'][1]
        if load_procedure == 'load':
            print('optimizing data')
            data_optimized = np.load('stats/data_optimized_{}.npy'.format(f.run_name))[()]
        elif load_procedure == 'reconstruct_before':
            stats = act_stats,fc2_stats, fc1_stats
            latent_placeholder = act_placeholder, fc2_placeholder, fc1_placeholder
            data_optimized = compute_optimized_examples(sess, stats,
                    f.train_batch_size, input_placeholder, latent_placeholder,
                    input_var, assign_op, recreate_op, data, latent_recreated,
                    recreate_loss, reinit_op, temp_recreated, temp_value)

            np.save('stats/data_optimized_{}.npy'.format(f.run_name), data_optimized)

        for i in range(f.epochs + 20):
            print('Epoch: {}'.format(i))
            for j in range(int(60000 / 64)):
                clas = j % 10
                if load_procedure in ['load', 'reconstruct_before']:
                    batch_idx = np.random.choice(len(data_optimized[clas]))
                    batch_x, batch_y = data_optimized[clas][batch_idx]
                elif load_procedure == 'reconstruct_fly':
                    batch_x, batch_y = sample_images(sess, stats, clas,
                            f.train_batch_size, input_placeholder,
                            latent_placeholder, input_var, assign_op,
                            recreate_op, data, latent_recreated, recreate_loss,
                            reinit_op, temp_recreated, temp_value)

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

        # post training, save statistics
        all_stats = {}
        all_stats['student_stats'] = compute_class_statistics(sess,
                '784-800-800-10/temp/div:0', inp, keep_inp, keep, data, 'temp:0', temp_value, stddev=True)
        all_stats['teacher_stats'] = compute_class_statistics(sess,
                '784-1200-1200-10/temp/div:0', inp, keep_inp, keep, data, 'temp_1:0', temp_value, stddev=True)
        np.save('stats/activation_stats_{}.npy'.format(f.run_name), all_stats)
        print('stats_saved')


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
