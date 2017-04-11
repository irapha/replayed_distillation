import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def sample_from_stats(means, cov, clas, batch_size, out_size):
    out_size = means[list(means.keys())[0]].shape[0]
    gauss = np.random.normal(size=(batch_size, out_size))
    pre_sftmx = means[clas] + np.matmul(gauss, cov[clas])
    return pre_sftmx

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

all_stats = np.load('stats/activation_stats_centralnorm_relumse_cov.npy')[()]

t_mean, t_sdev = all_stats['teacher_stats']

print(softmax(sample_from_stats(t_mean, t_sdev, 8, 1, 10)))
print(softmax(sample_from_stats(t_mean, t_sdev, 8, 1, 10)/90.0))
