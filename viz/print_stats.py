import numpy as np
import tensorflow as tf

np.set_printoptions(linewidth=200)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('run_name', '', 'The name of the experimental run')
flags.DEFINE_string('summary_folder', 'summaries/', 'Folder to save summaries, logs, stats, optimized_datasets')

all_stats = np.load(os.path.join(FLAGS.summary_folder, FLAGS.run_name, 'stats',
    'activation_stats_{}.npy'.format(FLAGS.run_name)))[()]

# TODO(sfenu3): if you modify what stats are being saved in compute_stats
# procedure, modify the line below too.
means, _, stddev, shape = all_stats[-1]

print('statistics for top layer shape=(?, {})'.format(shape))
for clas in range(len(means.keys())):
    print('class {} means'.format(clas))
    print(repr(means[clas]), end="\n\n")
    print('class {} stddev'.format(clas))
    print(repr(stddev[clas]), end="\n\n")


