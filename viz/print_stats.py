import os
import numpy as np
import tensorflow as tf

np.set_printoptions(linewidth=200)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('run_name', '', 'The name of the experimental run')
flags.DEFINE_string('summary_folder', 'summaries/', 'Folder where summaries, logs, stats, optimized_datasets are saved')

all_stats, _ = np.load(os.path.join(FLAGS.summary_folder, FLAGS.run_name, 'stats',
    'activation_stats_{}.npy'.format(FLAGS.run_name)))[()]

means, _, stddev, shape = all_stats[-1]

print('statistics for top layer shape=(?, {})'.format(shape))
for clas in range(len(means.keys())):
    print('class {}'.format(clas))
    print('means: {}'.format(repr(means[clas])))
    print('stddev: {}'.format(repr(stddev[clas])), end="\n\n")


