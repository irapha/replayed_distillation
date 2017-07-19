import os
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('run_name', '', 'The name of the experimental run')
flags.DEFINE_string('summary_folder', 'summaries/', 'Folder where summaries, logs, stats, optimized_datasets are saved')


def sample_from_stats(means, stddev, clas, batch_size, out_size):
    out_size = means[list(means.keys())[0]].shape[0]
    gauss = np.random.normal(size=(batch_size, out_size))
    pre_sftmx = means[clas] + np.matmul(gauss, stddev[clas])
    return pre_sftmx

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

all_stats, _ = np.load(os.path.join(FLAGS.summary_folder, FLAGS.run_name, 'stats',
    'activation_stats_{}.npy'.format(FLAGS.run_name)))[()]

# TODO(sfenu3): if you modify what stats are being saved in compute_stats
# procedure, modify the line below too.
means, _, stddev, _ = all_stats[-1]

print("sample from top layer statistics, class=8, batch_size=1, num_classes=10, temperature=1, softmax=True")
print(softmax(sample_from_stats(means, stddev, 8, 1, 10)))
print("sample from top layer statistics, class=8, batch_size=1, num_classes=10, temperature=90, softmax=True")
print(softmax(sample_from_stats(means, stddev, 8, 1, 10)/90.0))
