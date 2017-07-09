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
from . import optimization_objectives as o

from utils import ensure_dir_exists, merge_summary_list


def sample_from_stats(stats, clas, batch_size, out_size):
    means, cov = stats
    out_size = means[list(means.keys())[0]].shape[0]
    gauss = np.random.normal(size=(batch_size, out_size))
    pre_sftmx = means[clas] + np.matmul(gauss, cov[clas])
    #  pre_sftmx = means[clas] + np.multiply(gauss, cov[clas])
    return pre_sftmx
    #  return [softmax(x) for x in pre_sftmx]

def sample_images(sess, stats, clas, batch_size, input_placeholder,
        latent_placeholder, input_var, assign_op, recreate_op, data,
        latent_recreated, recreate_loss, reinit_op, temp_recreated,
        temp_rec_val, drop_dict):

    act_stats,fc2_stats, fc1_stats = stats
    act_placeholder, fc2_placeholder, fc1_placeholder = latent_placeholder

    # these hyper params can be tuned
    num_examples_per_median = 64
    noise = 0.1

    all_medians = []
    all_latents = []

    for i in range(1):
        act_latent = sample_from_stats(act_stats, clas, num_examples_per_median, 10)
        fc2_latent = sample_from_stats(fc2_stats, clas, num_examples_per_median, 1200)
        fc1_latent = sample_from_stats(fc1_stats, clas, num_examples_per_median, 1200)
        print('\tmedian: {}'.format(i))

        # currently setting noise to 0.1 and samples to 64, just bc that will make things easier rn
        sess.run(reinit_op)

        # TODO: remove
        #  input_kernels = [np.reshape(np.random.uniform(low=0.0, high=0.2, size=[28, 28]), [784]) for _ in range(num_examples_per_median)]
        input_kernels = [np.reshape(np.random.normal(0.15, 0.1, size=[28, 28]), [784]) for _ in range(num_examples_per_median)]

        sess.run(assign_op, feed_dict={input_placeholder: input_kernels})

        for _ in range(1000):
            _, los = sess.run([recreate_op, recreate_loss],
                    feed_dict={act_placeholder: act_latent,
                        fc2_placeholder: fc2_latent,
                        fc1_placeholder: fc1_latent,
                        temp_recreated: temp_rec_val})

        all_latents.extend(sess.run('const/div:0', feed_dict={temp_recreated: temp_rec_val}))
        all_medians = sess.run(input_var)

    return all_medians, all_latents


def run(sess, FLAGS, data):
    # things to do: load the graph and create its const version, create the
    # optimization ops (both the ones specific to the opt objective and the
    # input variable and things like it.
    # then run the optimization procedure and finally save the dataset.

    input_size, output_size = data.io_shape
    input_placeholder = tf.placeholder(tf.float32, [None, input_size], name='input_placeholder')
    input_var = tf.Variable(tf.zeros([f.train_batch_size, input_size]), name='recreated_imgs')
    sess.run(tf.variables_initializer([input_var], name='init_input'))
    assign_op = tf.assign(input_var, input_placeholder)

    _, layer_activations, feed_dicts = m.get(f.model).load_and_freeze_model(
            sess, input_var, f.model_meta, f.model_checkpoint, output_size)

    opt_obj = o.get(f.optimization_objective)(layer_activations)

    reinit_op = tf.variables_initializer(u.get_uninitted_vars(sess), name='reinit_op')
    sess.run(reinit_op)

    saver = tf.train.Saver(tf.global_variables())

    with sess.as_default():
        #  stats = act_stats,fc2_stats, fc1_stats
        #  latent_placeholder = act_placeholder, fc2_placeholder, fc1_placeholder

        # TODO where this goes?
        #  feed_dict = opt_obj.sample_from_stats(stats, feed_dict=feed_dicts['distill'])


        data_optimized = {clas: [] for clas in range(output_size)}
        for clas in range(output_size):
            print('optimizing examples for class: {}'.format(clas))
            for i in range(100):
                data_optimized[clas].append(sample_images(sess, stats, clas, train_batch_size,
                        input_placeholder, latent_placeholder, input_var, assign_op,
                        recreate_op, data, latent_recreated, recreate_loss, reinit_op,
                        temp_recreated, temp_rec_val, drop_dict))


        data_optimized = compute_optimized_examples(sess, stats,
                f.train_batch_size, input_placeholder, latent_placeholder,
                input_var, assign_op, recreate_op, data, latent_recreated,
                recreate_loss, reinit_op, temp_recreated, temp_value, drop_dict)

    data_dir = os.path.join(f.summary_folder, f.run_name, 'data')
    u.ensure_dir_exists(data_dir)
    data_file = os.path.join(data_dir, 'data_optmized_{}.npy'.format(f.run_name))
    np.save(data_file, data_optimized)
    print('data saved in {}'.format(data_file))
