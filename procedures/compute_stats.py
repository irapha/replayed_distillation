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
        layerwise_stats = [] # in order, from bottom to topmost activation

        for layer_activation, size in layer_activations:
            print('computing stats for {}'.format(layer_activation))
            stats = compute_layerwise_statistics(sess, layer_activation, size, inputs, data, feed_dicts)
            layerwise_stats.append(stats)

        if f.compute_graphwise_stats:
            print('computing stats for entire network')
            all_layers_stats = compute_graphwise_statistics(sess, layer_activations, inputs, data, feed_dicts)
        else:
            all_layers_stats = None

        all_stats = layerwise_stats, all_layers_stats

    stats_dir = os.path.join(f.summary_folder, f.run_name, 'stats')
    u.ensure_dir_exists(stats_dir)
    stats_file = os.path.join(stats_dir, 'activation_stats_{}.npy'.format(f.run_name))
    np.save(stats_file, all_stats)
    print('stats saved in {}'.format(stats_file))

def compute_layerwise_statistics(sess, tensor, size, inputs, data, feed_dicts):
    # compute activations for all examples in train set, organized by class
    means = {}
    stdev = {}
    cov   = {}

    for batch_x, batch_y in data.train_epoch_in_batches(10):
        batch_out = sess.run(tf.reshape(tensor, [-1, size]),
                feed_dict={**feed_dicts['distill'], inputs: batch_x})

        for act, y in zip(batch_out, batch_y):
            clas = np.where(y == 1)[0][0] # label in dataset
            if clas not in means:
                means[clas] = (np.zeros(act.shape), 0)
                stdev[clas] = (np.zeros(act.shape), np.zeros(act.shape), 0)
            means[clas] = (means[clas][0]+act, means[clas][1]+1)
            stdev[clas] = (stdev[clas][0]+act**2, stdev[clas][1]+act, stdev[clas][2]+1)
    
    for key in means.keys():
        means[key] = means[key][0]/means[key][1]

    for key in stdev.keys():
        stdev[key] = np.sqrt(stdev[key][0]/stdev[key][2] - (stdev[key][1]/stdev[key][2])**2)

    # TODO streaming covariance sketch from (Chi et al 2015)
    cov = None

    # save the shape too. will be needed for later.
    batch_x, _ = next(data.train_epoch_in_batches(50))
    batch_out = sess.run(tensor, feed_dict={**feed_dicts['distill'], inputs: batch_x})
    shape = (-1,) + np.shape(batch_out[0])[0:]

    return means, cov, stdev, shape


def compute_graphwise_statistics(sess, tensors, inputs, data, feed_dicts):
    # compute activations for all examples in train set, organized by class
    all_activations = {}
    activations, sizes = map(list, zip(*tensors))

    total_edges = sum(sizes)

    activation_graphs = {}
    classes = set([])

    for tensor, size in tensors:
        for batch_x, batch_y in data.train_epoch_in_batches(50):
            batch_out = sess.run(tf.reshape(tensor, [-1, size]),
                    feed_dict={**feed_dicts['distill'], inputs: batch_x})

            for act, y in zip(batch_out, batch_y):
                clas = np.where(y == 1)[0][0] # label in dataset
                classes |= set([clas])
                if tensor not in all_activations.keys():
                    all_activations[tensor] = {}

                if clas not in all_activations[tensor].keys():
                    all_activations[tensor][clas] = []

                all_activations[tensor][clas].append(act)

    # consolidate them each of the class' activations
    means = {t:{c: [] for c in classes}  for t, _ in tensors}

    for tensor, _ in tensors:
        for k, v in all_activations[tensor].items():
            means[tensor][k] = np.mean(v, axis=0)

    activation_graphs = {c: np.zeros(total_edges, total_edges) for c in classes}
    pairwise_activation_graphs = {c: [np.zeros((sizes[i], sizes[i+1])) for i in range(len(sizes)-1)] for c in classes}

    for clas in classes:
        for i in range(len(sizes) - 1):
            edge_a, edge_b = sizes[i], sizes[i+1]

            col_index += edge_a
            mean = np.outer(means[tensors[i][0]][clas], means[tensors[i+1][0]][clas])

            activation_graphs[clas][row_index:row_index+edge_a, col_index:col_index+edge_b] = mean
            row_index += edge_a

            pairwise_activation_graphs[clas][i][0:edge_a, edge_a:edge_a + edge_b] = mean


    tensor_reconstructions = {}
    pairwise_tensor_reconstructions = {c:[] for c in classes}

    for clas in classes:
        _, inverse_fourier_matrix = scipy.linalg.schur(activation_graphs[clas])
        fourier_matrix = scipy.linalg.inv(inverse_fourier_matrix)
        spectrum_coefficients = np.matrix(fourier_matrix) * np.matrix(activation_graphs[clas])
        subsampled_spectrum = np.zeros(spectrum_coefficients.shape)
        end_index = int(len(spectrum_coefficients) * downscale_factor)
        subsampled_spectrum[:end_index] = spectrum[:end_index]

        tensor_reconstructions[clas] = np.matrix(inverse_fourier_matrix) * np.matrix(subsampled_spectrum)

        for pair in range(len(pairwise_activation_graphs )):
            _, pairwise_inverse_fourier = scipy.linalg.schur(pairwise_activation_graphs[clas][pair])
            pairwise_fourier_matrix = scipy.linalg.inv(inverse_fourier_matrix)
            spectrum_coefficients = np.matrix(fourier_matrix) * np.matrix(pairwise_activation_graphs[clas][pair])
            subsampled_spectrum = np.zeros(spectrum_coefficients.shape)

            end_index = int(len(spetrum_coefficients) * downscale_factor)
            subsampled_spectrum[:end_index] = spectrum[:end_index]

            pairwise_tensor_reconstructions[clas].append(np.matrix(inverse_fourier_matrix) * np.matrix(pairwise_activation_graphs[clas][pair]))

    return reconstructions, pairwise_reconstructions, shape
