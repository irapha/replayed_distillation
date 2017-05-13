"""
This training procedure will train a student network from scratch using
replayed data sampled from a teacher model, and the sampled activations from
that teacher.
"""
import numpy as np
import os
import sys
import tensorflow as tf
#import cv2
import scipy.stats as st
import scipy

import models as m
import utils as u

from utils import ensure_dir_exists

MODEL_META = 'summaries/train_lenet_realinit/checkpoint/lenet-8000.meta'
MODEL_CHECKPOINT = 'summaries/train_lenet_realinit/checkpoint/lenet-8000'

# TODOS:
# - [done] see how relu + l2 reconstruction look like
# - try pruning og model (with og weights) but using optimized examples.
# - [done] try regenerating data without median calculation. Should be quick.
# - [done] use cov matrix again. needs to be stddev tho
# - [done] maybe keep stats for middle of model. then feedforward to get activations
#          for distilling, and optimize input on middle sample.
# - [done] keep stats for all layers actually, then get MSE of each of those, and
#          reconstruct input on all of them.
# - [done] rerun this new all layers distillation on mnist to see how well the student does
# - [done] fix dropout neurons at every reconstruction step. i think it will
#          better use the specific neuron pathways backwards.
# - spectral optimization objective
# - make it work with convolutions
# - see how well it does with just initializing student using matrix factorization techniques

# CURRENTLY WORKING IN TODO CONV.

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

def get_dropout_filter(shape1, shape2, keep_prob):
    return np.random.binomial(1, keep_prob, size=[shape1, shape2])

def compute_class_statistics(sess, act_tensor, flat_shape, inp, data, temp, temp_val, stddev=False):
    all_activations = {}
    for batch_x, batch_y in data.train_epoch_in_batches(50, size=32):
        #  batch_size = len(batch_x)
        #  batch_out = sess.run('labels_sftmx/Reshape_1:0',
        #  print(np.shape(batch_x))
        ten = tf.get_default_graph().get_tensor_by_name(act_tensor+':0')
        batch_out = sess.run(tf.reshape(ten, [-1, flat_shape]),
                feed_dict={inp: batch_x,
                    #  keep_inp: 1.0, keep: 1.0,
                    temp: temp_val})

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
        # TODO CONV: probably flatten before cholesky, then probably unflatten when feeding as placeholder
        # should be done. im flatening above^ at sess.run. then we store the
        # flatenned stats. those can be directly sampled and fed as flat to
        # graph.
        # TODO CONV: figure out a way to use cov and chol. apparently conv1 is not positive definite...
        if 'conv' not in act_tensor:
            cov[k] = np.linalg.cholesky(np.cov(np.transpose(v)))
        stdev[k] = np.sqrt(np.var(v, axis=0))

    if stddev:
        return means, cov, stdev
    # TODO CONV: s/stdev/cov -> as soon as you fix the cov todo above^
    if 'conv' in act_tensor:
        return means, stdev
    return means, cov

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def sample_from_stats(stats, clas, batch_size, out_size, is_conv=False):
    means, cov = stats
    out_size = means[list(means.keys())[0]].shape[0]
    gauss = np.random.normal(size=(batch_size, out_size))
    # TODO CONV: this cov is actually stdev so we're doing element wise gaussian. revert to matrix once we use cov again
    if is_conv:
        pre_sftmx = means[clas] + np.multiply(gauss, cov[clas])
    else:
        pre_sftmx = means[clas] + np.matmul(gauss, cov[clas])
    return pre_sftmx
    #  pre_sftmx = means[clas] + np.multiply(gauss, cov[clas])
    #  return [softmax(x) for x in pre_sftmx]

#def gkern(size=28, sig=4, noise=0.1):
#    g = cv2.getGaussianKernel(size, sig)
#    gi = cv2.getGaussianKernel(size, sig).T
#    normd = np.matmul(g, gi)
#    normd = (1.0 - noise) * (normd / normd.max())
#    normd += noise * np.random.uniform(size=[size, size])
#    return normd

METHOD = ['onehot', 'onesample', 'manysample'][2]

def sample_images(sess, stats, clas, batch_size, input_placeholder,
        latent_placeholder, input_var, assign_op, recreate_op, data,
        latent_recreated, recreate_loss, reinit_op, temp_recreated,
        temp_rec_val, drop_dict):

    #  act_stats,fc2_stats, fc1_stats = stats
    conv1_stats, conv2_stats, fc1_stats, fc2_stats, fc3_stats = stats
    #  act_placeholder, fc2_placeholder, fc1_placeholder = latent_placeholder
    conv1_placeholder, conv2_placeholder, fc1_placeholder, fc2_placeholder, fc3_placeholder = latent_placeholder

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
            # TODO CONV: fix these shapes probs
            # should be done.
            conv1_latent = sample_from_stats(conv1_stats, clas, num_examples_per_median, 28*28*6, is_conv=True)
            conv2_latent = sample_from_stats(conv2_stats, clas, num_examples_per_median, 10*10*16, is_conv=True)
            fc1_latent = sample_from_stats(fc1_stats, clas, num_examples_per_median, 120)
            fc2_latent = sample_from_stats(fc2_stats, clas, num_examples_per_median, 84)
            fc3_latent = sample_from_stats(fc3_stats, clas, num_examples_per_median, 10)
        print('\tmedian: {}'.format(i))

        # currently setting noise to 0.1 and samples to 64, just bc that will make things easier rn
        sess.run(reinit_op)
        #  input_kernels = [np.reshape(gkern(noise=noise), [784]) for _ in range(num_examples_per_median)]

        # TODO: remove
        #  input_kernels = [np.reshape(np.random.uniform(low=0.0, high=0.2, size=[28, 28]), [784]) for _ in range(num_examples_per_median)]
        input_kernels = [np.random.normal(0.15, 0.1, size=[32, 32, 1]) for _ in range(num_examples_per_median)]

        sess.run(assign_op, feed_dict={input_placeholder: input_kernels})

        # TODO NIPS: use drop_dict
        #  sess.run(drop_dict['assign_drop_inp_op'],
                #  feed_dict={drop_dict['drop_inp_place']: get_dropout_filter(batch_size, 784, 0.8)})
        #  sess.run(drop_dict['assign_drop_fc1_op'],
                #  feed_dict={drop_dict['drop_fc1_place']: get_dropout_filter(batch_size, 1200, 0.5)})
        #  sess.run(drop_dict['assign_drop_fc2_op'],
                #  feed_dict={drop_dict['drop_fc2_place']: get_dropout_filter(batch_size, 1200, 0.5)})

        for _ in range(1000):
            _, los = sess.run([recreate_op, recreate_loss],
                    feed_dict={
                        conv1_placeholder: conv1_latent,
                        conv2_placeholder: conv2_latent,
                        fc1_placeholder: fc1_latent,
                        fc2_placeholder: fc2_latent,
                        fc3_placeholder: fc3_latent,
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
        temp_recreated, temp_rec_val, drop_dict):
    opt = {}
    for clas in range(10):
        print('clas: {}'.format(clas))
        if clas not in opt:
            opt[clas] = []
        for i in range(100):
            opt[clas].append(sample_images(sess, stats, clas, train_batch_size,
                    input_placeholder, latent_placeholder, input_var, assign_op,
                    recreate_op, data, latent_recreated, recreate_loss, reinit_op,
                    temp_recreated, temp_rec_val, drop_dict))
    return opt


def run(sess, f, data, placeholders, train_step, summary_op, summary_op_evaldistill):
    #  inp, labels, keep_inp, keep, temp, labels_temp, labels_evaldistill = placeholders
    #  inp, labels, temp, labels_temp = placeholders
    inp, labels, temp, labels_temp, labels_evaldistill = placeholders

    # create same model with constants instead of vars. And a Variable as input
    print('creating const graph')
    with tf.variable_scope('const'):
        # TODO CONV: fix shapes for these conv placeholders. probably will involve modifying sampling bc of reshaping into 4 dims
        # should be done.
        fc3_placeholder = tf.placeholder(tf.float32, [None, 10], name='fc3_placeholder')
        fc2_placeholder = tf.placeholder(tf.float32, [None, 84], name='fc2_placeholder')
        fc1_placeholder = tf.placeholder(tf.float32, [None, 120], name='fc1_placeholder')
        conv2_placeholder = tf.reshape(tf.placeholder(tf.float32, [None, 10*10*16], name='conv2_placeholder'), [-1, 10, 10, 16])
        conv1_placeholder = tf.reshape(tf.placeholder(tf.float32, [None, 28*28*6], name='conv1_placeholder'), [-1, 28, 28, 6])

        input_placeholder = tf.placeholder(tf.float32, [None, 32, 32, 1], name='input_placeholder')
        input_var = tf.Variable(tf.zeros([f.train_batch_size, 32, 32, 1]), name='recreated_imgs')
        temp_recreated = tf.placeholder(tf.float32, name='temp_recreated')
        sess.run(tf.variables_initializer([input_var], name='init_input'))
        assign_op = tf.assign(input_var, input_placeholder)

        drop_dict = {}
        #  drop_dict['drop_inp_place'] = tf.placeholder(tf.float32, [f.train_batch_size, 784], name='drop_inp_place')
        #  drop_dict['drop_inp_var'] = tf.Variable(tf.zeros([f.train_batch_size, 784]), name='drop_inp_var')
        #  drop_dict['assign_drop_inp_op'] = tf.assign(drop_dict['drop_inp_var'], drop_dict['drop_inp_place'])

        #  drop_dict['drop_fc1_place'] = tf.placeholder(tf.float32, [f.train_batch_size, 1200], name='drop_fc1_place')
        #  drop_dict['drop_fc1_var'] = tf.Variable(tf.zeros([f.train_batch_size, 1200]), name='drop_fc1_var')
        #  drop_dict['assign_drop_fc1_op'] = tf.assign(drop_dict['drop_fc1_var'], drop_dict['drop_fc1_place'])

        #  drop_dict['drop_fc2_place'] = tf.placeholder(tf.float32, [f.train_batch_size, 1200], name='drop_fc2_place')
        #  drop_dict['drop_fc2_var'] = tf.Variable(tf.zeros([f.train_batch_size, 1200]), name='drop_fc2_var')
        #  drop_dict['assign_drop_fc2_op'] = tf.assign(drop_dict['drop_fc2_var'], drop_dict['drop_fc2_place'])

        latent_recreated = tf.div(m.get('lenet').create_constant_model(sess, input_var, drop_dict), temp_recreated)
        with tf.variable_scope('xent_recreated'):
            if METHOD == 'onehot':
                sft = latent_placeholder
            else:
                #  sft = tf.nn.sigmoid(latent_placeholder)
                #  sft = tf.nn.softmax(latent_placeholder)
                conv1_sft = tf.nn.relu(conv1_placeholder)
                conv2_sft = tf.nn.relu(conv2_placeholder)
                fc1_sft = tf.nn.relu(fc1_placeholder)
                fc2_sft = tf.nn.relu(fc2_placeholder)
                fc3_sft = tf.nn.relu(fc3_placeholder)
                #  sft = latent_placeholder
                # TODO CONV: replace this optimization objective with all conv layers, properly rescaled to the layer shapes.
                # should be done. just need to verify names and check that this works
            recreate_loss = (
                    ((1.0/(28*28*6)) *
                        tf.reduce_mean(
                            tf.pow((conv1_sft - tf.nn.relu(
                                tf.get_default_graph().get_tensor_by_name('const/lenet-5_const/conv1/add:0')
                                )), 2)
                            )) +
                    ((1.0/(10*10*16)) *
                        tf.reduce_mean(
                            tf.pow((conv2_sft - tf.nn.relu(
                                tf.get_default_graph().get_tensor_by_name('const/lenet-5_const/conv2/add:0')
                                )), 2)
                            )) +
                    ((1.0/120) *
                        tf.reduce_mean(
                            tf.pow((fc1_sft - tf.nn.relu(
                                tf.get_default_graph().get_tensor_by_name('const/lenet-5_const/fc1/add:0')
                                )), 2)
                            )) +
                    ((1.0/84) *
                        tf.reduce_mean(
                            tf.pow((fc2_sft - tf.nn.relu(
                                tf.get_default_graph().get_tensor_by_name('const/lenet-5_const/fc2/add:0')
                                )), 2)
                            )) +
                    ((1.0/10) *
                        tf.reduce_mean(
                            tf.pow((fc3_sft - tf.nn.relu(
                                tf.get_default_graph().get_tensor_by_name('const/div:0')
                                )), 2)
                            ))
                    )
        with tf.variable_scope('opt_recreated'):
            recreate_op = tf.train.AdamOptimizer(learning_rate=0.09).minimize(recreate_loss) # lr was 0.07

        reinit_op = tf.variables_initializer(u.get_uninitted_vars(sess), name='reinit_op')
        sess.run(reinit_op)

    print('all graphs created')

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
        print('computing stats 1')
        conv1_stats = compute_class_statistics(sess, 'lenet-5/conv1/add', 28*28*6, inp, data, 'temp_1:0', temp_value)
        print('computing stats 2')
        conv2_stats = compute_class_statistics(sess, 'lenet-5/conv2/add', 10*10*16, inp, data, 'temp_1:0', temp_value)
        print('computing stats 3')
        fc1_stats = compute_class_statistics(sess, 'lenet-5/fc1/add', 120, inp, data, 'temp_1:0', temp_value)
        print('computing stats 4')
        fc2_stats = compute_class_statistics(sess, 'lenet-5/fc2/add', 84, inp, data, 'temp_1:0', temp_value)
        print('computing stats 5')
        fc3_stats = compute_class_statistics(sess, 'lenet-5/temp/div', 10, inp, data, 'temp_1:0', temp_value)
        print('all stats computed')
        load_procedure = ['load', 'reconstruct_before', 'reconstruct_fly'][1]
        if load_procedure == 'load':
            print('loading optimizing data')
            data_optimized = np.load('stats/data_optimized_{}.npy'.format(f.run_name))[()]
        elif load_procedure == 'reconstruct_before':
            print('reconstructing optimized data')
            stats = conv1_stats, conv2_stats, fc1_stats, fc2_stats, fc3_stats
            #  latent_placeholder = act_placeholder, fc2_placeholder, fc1_placeholder
            latent_placeholder = conv1_placeholder, conv2_placeholder, fc1_placeholder, fc2_placeholder, fc3_placeholder
            data_optimized = compute_optimized_examples(sess, stats,
                    f.train_batch_size, input_placeholder, latent_placeholder,
                    input_var, assign_op, recreate_op, data, latent_recreated,
                    recreate_loss, reinit_op, temp_recreated, temp_value, drop_dict)

            np.save('stats/data_optimized_{}.npy'.format(f.run_name), data_optimized)
            print('optimized data saved: stats/data_optimized_{}.npy'.format(f.run_name))

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
                            reinit_op, temp_recreated, temp_value, drop_dict)

                summary, _ = sess.run([summary_op_evaldistill, train_step],
                        feed_dict={inp: batch_x,
                            labels_evaldistill: batch_y,
                            #  keep_inp: 1.0, keep: 1.0,
                            'temp_1:0': 8.0, temp: 8.0})

                trainbatch_writer.add_summary(summary, global_step)

                if global_step % f.eval_interval == 0:
                    # eval test
                    summaries = []
                    for test_batch_x, test_batch_y in data.test_epoch_in_batches(f.test_batch_size, size=32):
                        summary = sess.run(summary_op_evaldistill,
                                feed_dict={inp: test_batch_x,
                                    labels_evaldistill: test_batch_y,
                                    #  keep_inp: 1.0, keep: 1.0,
                                    'temp_1:0': 1.0, temp: 1.0})
                        summaries.append(summary)
                    test_writer.add_summary(merge_summary_list(summaries, True), global_step)

                    # eval train
                    summaries = []
                    for train_batch_x, train_batch_y in data.train_epoch_in_batches(f.train_batch_size, size=32):
                        summary = sess.run(summary_op_evaldistill,
                                feed_dict={inp: train_batch_x,
                                    labels_evaldistill: train_batch_y,
                                    #  keep_inp: 1.0, keep: 1.0,
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
                'lenet-5_half/temp/div:0', inp, data, 'temp:0', temp_value, stddev=True)
        all_stats['teacher_stats'] = compute_class_statistics(sess,
                'lenet-5/temp/div:0', inp, data, 'temp_1:0', temp_value, stddev=True)
        np.save('stats/activation_stats_{}.npy'.format(f.run_name), all_stats)
        print('stats saved : stats/activation_stats_{}.npy'.format(f.run_name))


def create_placeholders(sess, input_size, output_size, _):
    new_saver = tf.train.import_meta_graph(MODEL_META)
    new_saver.restore(sess, MODEL_CHECKPOINT)

    inp = tf.get_collection('input')[0]
    out = tf.get_collection('output')[0]
    #  keep_inp = tf.get_collection('keep_inp')[0]
    #  keep = tf.get_collection('keep')[0]
    labels_temp = tf.get_collection('labels_temp')[0]

    with tf.variable_scope('labels_sftmx'):
        labels = tf.nn.softmax(out)

    with tf.variable_scope('stoppp'):
        labels = tf.stop_gradient(labels)

    return inp, labels, None, None, labels_temp
