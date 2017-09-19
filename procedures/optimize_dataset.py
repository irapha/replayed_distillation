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

from . import _optimization_objectives as o


def run(sess, f, data):
    input_size, output_size = data.io_shape
    input_placeholder = tf.placeholder(tf.float32, [None, input_size], name='input_placeholder')
    input_var = tf.Variable(tf.zeros([f.train_batch_size, input_size]), name='recreated_imgs')
    sess.run(tf.variables_initializer([input_var], name='init_input'))
    # this op is used at every new optimized batch, to initialize the input tf.Variable to random noise.
    assign_op = tf.assign(input_var, input_placeholder)

    # create "frozen" model, where each variable is now a constant.
    # the only thing being updated at every train step is the input tf.Variable.
    outputs, layer_activations, feed_dicts, dropout_filters = m.get(f.model).load_and_freeze_model(
            sess, input_var, f.model_meta, f.model_checkpoint, f.train_batch_size, output_size)

    # create ops specific to the optimization objective
    # (top_layer, all_layers, all_layers_dropout, spectral_all_layers, spectral_layer_pairs)
    opt_obj = o.get(f.optimization_objective)(layer_activations, float(f.lr))
    # this created class will also be used to create the feed_dicts we need for
    # the optimization objective at every train step.

    # this op is used to restart the graph (e.g.: all variables that the optimizer uses)
    reinit_op = tf.variables_initializer(u.get_uninitted_vars(sess), name='reinit_op')
    sess.run(reinit_op)

    saver = tf.train.Saver(tf.global_variables())

    with sess.as_default():
        # load stats we saved from compute_stats procedure.
        # this assumes that run_name is the same.
        stats = np.load(os.path.join(f.summary_folder, f.run_name, 'stats',
                                     'activation_stats_{}.npy'.format(f.run_name)))[()]

        #  data_optimized = {clas: [] for clas in range(output_size)}
        num_classes = output_size // 2 if f.loss == 'attrxent' else output_size
        for clas in range(num_classes):
            print('optimizing examples for class: {}'.format(clas))
            # creating 100 batches for this class
            # the data is saved as a list of batches, each with batch_size = f.train_batch_size
            # each batch is saved as a tuple of (np.array of images, np.array of latent outputs)
            for i in range(40):
                print('batch {}/40'.format(i), end='\r')
                # reinitialize graph
                sess.run(reinit_op)

                # reinitialize dropout filters. Only all_layers_dropout does
                # something fancy here. The other optimization_objectives just
                # rescale each layer by keep_prob (which should be done in a
                # model trained with dropout being used for inference)
                opt_obj.reinitialize_dropout_filters(sess, dropout_filters)

                # reinitialize input tf.Variable to random noise
                # use this for mnist:
                #  input_kernels = [np.random.normal(0.15, 0.1, size=[input_size]) for _ in range(f.train_batch_size)]
                # use this for celeba:
                input_kernels = [np.random.normal(0.44, 0.29, size=[input_size]) for _ in range(f.train_batch_size)]
                sess.run(assign_op, feed_dict={input_placeholder: input_kernels})

                # initialize feed_dict with whatever samples from stats the optimization objective needs
                optimize_feed_dict = opt_obj.sample_from_stats(stats, clas, f.train_batch_size, feed_dicts=feed_dicts)

                ### REMOVE
                #  optimized_inputs = sess.run(input_var)
                #  print('min: {}'.format(min(optimized_inputs[0])))
                #  print('max: {}'.format(max(optimized_inputs[0])))
                #  print('mean: {}'.format(np.mean(np.reshape(optimized_inputs, (-1,)))))
                #  print('std: {}'.format(np.std(np.reshape(optimized_inputs, (-1,)))))

                # create blur op
                blur_kernel = tf.constant([
                    [[[1/16, 0, 0], [0, 1/16, 0], [0, 0, 1/16]],
                     [[1/8 , 0, 0], [0, 1/8 , 0], [0, 0, 1/8 ]],
                     [[1/16, 0 ,0], [0, 1/16, 0], [0, 0, 1/16]]],
                    [[[1/8 , 0, 0], [0, 1/8 , 0], [0, 0, 1/8 ]],
                     [[1/4 , 0, 0], [0, 1/4 , 0], [0, 0, 1/4 ]],
                     [[1/8 , 0 ,0], [0, 1/8 , 0], [0, 0, 1/8 ]]],
                    [[[1/16, 0, 0], [0, 1/16, 0], [0, 0, 1/16]],
                     [[1/8 , 0, 0], [0, 1/8 , 0], [0, 0, 1/8 ]],
                     [[1/16, 0 ,0], [0, 1/16, 0], [0, 0, 1/16]]]])
                padded_input = tf.pad(tf.reshape(input_var, [64, 224, 224, 3]), [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC')
                blurred_inputs = tf.nn.conv2d(padded_input, blur_kernel, strides=[1,1,1,1], padding='VALID')
                blurred_clipped_inputs = tf.clip_by_value(blurred_inputs, 0.0, 1.0)
                blur_clip_op = tf.assign(input_var, tf.reshape(blurred_clipped_inputs, [64, -1]))
                ### REMOVE END

                # the actual optimization step, where we backprop to the input tf.Variable
                #  for _ in range(1000):
                for _ in range(1000):
                    _ = sess.run(opt_obj.recreate_op, feed_dict=optimize_feed_dict)
                    ### REMOVE
                    # blur the input and clip values to [0,1]
                    sess.run(blur_clip_op)
                    ### REMOVE END

                optimized_inputs = sess.run(input_var)
                optimized_outputs = [sess.run(outputs, feed_dict=feed_dicts['distill'])]
                #  data_optimized[clas].append((optimized_inputs, optimized_outputs))

                ### REMOVE
                #  print('vvv')
                #  import cv2
                #  import sys
                #  from viz.view import reshape_to_row

                #  print('min: {}'.format(min(optimized_inputs[0])))
                #  print('max: {}'.format(max(optimized_inputs[0])))
                #  print('mean: {}'.format(np.mean(np.reshape(optimized_inputs, (-1,)))))
                #  print('std: {}'.format(np.std(np.reshape(optimized_inputs, (-1,)))))

                #  means = [np.squeeze(np.mean(optimized_inputs, axis=0))]
                #  cv2.imshow('means.png', reshape_to_row(np.array(means), side=224, rgb=True))
                #  cv2.waitKey(0)
                #  sys.exit(-1)
                ### REMOVE END

                # save this class' optimized data. keeping everything in memory is too much.
                data_dir = os.path.join(f.summary_folder, f.run_name, 'data')
                u.ensure_dir_exists(data_dir)
                data_file = os.path.join(data_dir, 'data_optimized_{}_{}_{}_{}.npy'.format(f.optimization_objective, f.run_name, clas, i))
                np.save(data_file, (optimized_inputs, optimized_outputs))

    data_dir = os.path.join(f.summary_folder, f.run_name, 'data')
    u.ensure_dir_exists(data_dir)
    data_file = os.path.join(data_dir, 'data_optimized_{}_{}_<clas>_<batch>.npy'.format(f.optimization_objective, f.run_name))
    #  np.save(data_file, data_optimized)
    print('data saved in {}'.format(data_file))
