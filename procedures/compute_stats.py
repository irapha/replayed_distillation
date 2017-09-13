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
import scipy.sparse


def run(sess, f, data):
    # load graph
    _, output_size = data.io_shape
    inputs, _, layer_activations, feed_dicts = m.get(f.model).load_model(sess, f.model_meta, f.model_checkpoint, output_size)

    with sess.as_default():
        layerwise_stats = [] # in order, from bottom to topmost activation
        for layer_activation, size in layer_activations:
            print('computing stats for {}'.format(layer_activation))
            stats = compute_layerwise_statistics(sess, layer_activation, size, inputs, data, feed_dicts, f.loss)
            layerwise_stats.append(stats)
        all_layers_stats = None

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

def compute_layerwise_statistics(sess, tensor, size, inputs, data, feed_dicts, lossform):
    # compute activations for all examples in train set, organized by class
    means = {}
    stdev = {}
    cov   = {}

    for batch_x, batch_y in data.train_epoch_in_batches(10):
        batch_out = sess.run(tf.reshape(tensor, [-1, size]),
                feed_dict={**feed_dicts['distill'], inputs: batch_x})

        for act, y in zip(batch_out, batch_y):
            if lossform == "attrxent":
                # many classes
                classes = np.where(np.where(np.reshape(y, (size//2, 2)) == 1)[1] == 1)[0]
            else:
                classes = [np.where(y == 1)[0][0]] # label in dataset

            for clas in classes:
                if clas not in means:
                    means[clas] = (np.zeros(act.shape), 0)
                    stdev[clas] = (np.zeros(act.shape), np.zeros(act.shape), 0)
                    cov[clas]   = (np.zeros(act.shape), np.zeros((act.shape[0], 1)), np.zeros((act.shape[0], act.shape[0])), 0)

                means[clas] = (means[clas][0]+act, means[clas][1]+1)
                stdev[clas] = (stdev[clas][0]+act**2, stdev[clas][1]+act, stdev[clas][2]+1)

                n = cov[clas][-1] + 1
                dx = act - cov[clas][0]
                meanx = cov[clas][0] + dx/n
                meany = cov[clas][1] + (np.array([act]).transpose() - cov[clas][1])/n
                C = cov[clas][2] + dx * (np.array([act]).transpose() - meany)

                cov[clas] = (meanx, meany, C, n)

    for key in means.keys():
        means[key] = means[key][0]/means[key][1]

    for key in stdev.keys():
        stdev[key] = np.sqrt(stdev[key][0]/stdev[key][2] - (stdev[key][1]/stdev[key][2])**2)

    for key in cov.keys():
        cov[key] = cov[key][2]/cov[key][-1]

    # save the shape too. will be needed for later.
    batch_x, _ = next(data.train_epoch_in_batches(8))
    batch_out = sess.run(tensor, feed_dict={**feed_dicts['distill'], inputs: batch_x})
    shape = (-1,) + np.shape(batch_out[0])[0:]

    return means, cov, stdev, shape


def compute_graphwise_statistics(sess, tensors, inputs, data, feed_dicts):
    # compute activations for all examples in train set, organized by class

    activation_means   = {}
    activations, sizes = map(list, zip(*tensors))

    total_edges = sum(sizes)

    classes = set([])

    for tensor, size in tensors:
        for batch_x, batch_y in data.train_epoch_in_batches(8):
            batch_out = sess.run(tf.reshape(tensor, [-1, size]),
                    feed_dict={**feed_dicts['distill'], inputs: batch_x})

            for act, y in zip(batch_out, batch_y):
                clas = np.where(y == 1)[0][0] # label in dataset
                classes |= set([clas])

                if tensor not in activation_means.keys():
                    activation_means[tensor] = {}

                if clas not in activation_means[tensor].keys():
                    activation_means[tensor][clas] = (np.zeros(act.shape), 0)

                activation_means[tensor][clas] = (activation_means[tensor][clas][0]+act, activation_means[tensor][clas][1] + 1)

    # normalize accumulated sums
    for tensor in activation_means.keys():
        for clas in activation_means[tensor].keys():
            activation_means[tensor][clas] = activation_means[tensor][clas][0] / activation_means[tensor][clas][1]

    # consolidate them each of the class' activations
    tensor_reconstructions = {c:[] for c in classes}

    for clas in classes:
        col_index = 0
        row_index = 0
        activation_graph = scipy.sparse.lil_matrix((total_edges, total_edges))

        for i in range(len(sizes) - 1):
            edge_a, edge_b = sizes[i], sizes[i+1]

            col_index += edge_a
            print("Tensor sizes: ")
            print(activation_means[tensors[i][0]][clas].shape)
            print(activation_means[tensors[i+1][0]][clas].shape)
            mean = np.outer(activation_means[tensors[i][0]][clas], activation_means[tensors[i+1][0]][clas])
            activation_graph[row_index:row_index+edge_a, col_index:col_index+edge_b] = mean
            row_index += edge_a

        activation_graph = activation_graph.tocsr()

        # TODO switch to lanczos
        n_vals = int(len(activation_graph)* downscale_factor)
        eigvals, eigvecs = scipy.sparse.linalg.eigs(activation_graph, k=n_vals)
        reconstructions = np.dot(eigvecs, eigvals[:,np.newaxis] * eigvecs.T)

        col_index = 0
        row_index = 0

        for i in range(len(sizes)-1):
            edge_a, edge_b = sizes[i], sizes[i+1]
            col_index += edge_a
            mean = reconstructions[row_index:row_index + edge_a, col_index:col_index+edge_b]
            row_index += edge_b
            tensor_reconstructions[clas].append(mean)


    batch_x, _ = next(data.train_epoch_in_batches(8))
    batch_out = sess.run(tensor, feed_dict={**feed_dicts['distill'], inputs: batch_x})
    shape = (-1,) + np.shape(batch_out[0])[0:]

    return tensor_reconstructions, shape
