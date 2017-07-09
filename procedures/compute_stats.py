"""
This training procedure will train a student network from scratch using
replayed data sampled from a teacher model, and the sampled activations from
that teacher.
"""
import numpy as np
import os
import tensorflow as tf
import models as m
import utils as u


def run(sess, f, data):
    # load graph
    _, output_size = data.io_shape
    inputs, _, layer_activations, feed_dicts = m.get(f.model).load_model(sess, f.model_meta, f.model_checkpoint, output_size)

    with sess.as_default():
        all_stats = [] # in order, from bottom to topmost activation
        for layer_activation, _ in layer_activations:
            print('computing stats for {}'.format(layer_activation))
            stats = compute_class_statistics(sess, layer_activation, inputs, data, feed_dicts)
            all_stats.append(stats)

    stats_dir = os.path.join(f.summary_folder, f.run_name, 'stats')
    u.ensure_dir_exists(stats_dir)
    stats_file = os.path.join(stats_dir, 'activation_stats_{}.npy'.format(f.run_name))
    np.save(stats_file, all_stats)
    print('stats saved in {}'.format(stats_file))

def compute_class_statistics(sess, tensor, inputs, data, feed_dicts):
    # compute activations for all examples in train set, organized by class
    all_activations = {}
    for batch_x, batch_y in data.train_epoch_in_batches(50):
        batch_out = sess.run(tensor,
                feed_dict={**feed_dicts['distill'], inputs: batch_x})

        for act, y in zip(batch_out, batch_y):
            clas = np.where(y == 1)[0][0] # label in dataset
            if clas not in all_activations:
                all_activations[clas] = []
            all_activations[clas].append(act)

    # consolidate them each of the class' activations
    means = {}
    cov = {}
    stdev = {}

    # TODO(sfenu3): compute and save spectral statistics
    for k, v in all_activations.items():
        means[k] = np.mean(v, axis=0)
        cov[k] = np.linalg.cholesky(np.cov(np.transpose(v)))
        stdev[k] = np.sqrt(np.var(v, axis=0))

    return means, cov, stdev
