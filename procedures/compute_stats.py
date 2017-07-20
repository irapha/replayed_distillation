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
        
        print('computing stats for entire network')
        all_layers_stats = compute_graphwise_statistics(sess, layer_activations, inputs, data, feed_dicts)

        all_stats = layerwise_stats, all_layers_stats

    stats_dir = os.path.join(f.summary_folder, f.run_name, 'stats')
    u.ensure_dir_exists(stats_dir)
    stats_file = os.path.join(stats_dir, 'activation_stats_{}.npy'.format(f.run_name))
    np.save(stats_file, all_stats)
    print('stats saved in {}'.format(stats_file))

def compute_layerwise_statistics(sess, tensor, size, inputs, data, feed_dicts):
    # compute activations for all examples in train set, organized by class
    all_activations = {}
    for batch_x, batch_y in data.train_epoch_in_batches(50):
        batch_out = sess.run(tf.reshape(tensor, [-1, size]),
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

    for k, v in all_activations.items():
        means[k] = np.mean(v, axis=0)
        # TODO(rapha): figure out a way to use cov for convolutional models
        # apparently conv1 is not positive definite. Our workaround is to
        # just use (and sample from) stddev when its a problem.
        if cov is not None:
            try:
                cov[k] = np.linalg.cholesky(np.cov(np.transpose(v)))
            except np.linalg.linalg.LinAlgError:
                cov = None
        stdev[k] = np.sqrt(np.var(v, axis=0))

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
