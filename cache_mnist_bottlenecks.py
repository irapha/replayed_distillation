# loads model and checkpoint, runs mnist through it, caches the output with a
# temperatured softmax
# python cache_mnist_bottlenecks.py --dataset=mnist --model=hinton1200 --checkpoint_dir summaries/hinton1200_mnist/checkpoint

import tensorflow as tf
import numpy as np

import models as m
import datasets as d
import utils as u

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', '', 'ImageNet or MNIST')
flags.DEFINE_string('model', '', 'hinton1200, hinton800, lenet300100, lenet5, alexnet, vgg16')
flags.DEFINE_string('bottleneck_file', '', 'where the bottlenecks will be saved')
flags.DEFINE_string('checkpoint_dir', '', 'where the model will be restored from')


if __name__ == '__main__':
    # create graph
    input_size, output_size = d.get_io_size(FLAGS.dataset)
    inp, _ = u.create_placeholders(input_size, output_size)

    keep_inp, keep, temp, labels_temp = u.create_optional_params()
    out = m.get(FLAGS.model).create_model(inp, output_size, keep_inp, keep, temp)

    # restore checkpoint
    saver = tf.train.Saver()

    # initialize dataset interface
    data = d.get(FLAGS.dataset)

    # initialize session
    sess = tf.Session(u.get_sess_config(use_gpu=True))

    with sess.as_default():
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))

        all_bottlenecks = []

        for img in data.og.train.images:
            bottleneck = sess.run(out,
                    feed_dict={
                        inp: [img],
                        keep_inp: 1.0, keep: 1.0, temp: 1.0})
            all_bottlenecks.extend(bottleneck)

    # save bottleneck_file
    all_bottlenecks = np.array(all_bottlenecks)
    with open(FLAGS.bottleneck_file, 'w') as f:
        np.save(f, all_bottlenecks)
