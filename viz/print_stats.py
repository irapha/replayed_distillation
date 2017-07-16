import numpy as np

np.set_printoptions(linewidth=200)

all_stats = np.load('stats/activation_stats_replay_drop_rescaleonly.npy')[()]

s_mean, s_sdev = all_stats['student_stats']
t_mean, t_sdev = all_stats['teacher_stats']

print('student mean\nteacher mean\nstudent stddev\nteacher stddev\n')

for clas in range(10):
    for a in [s_mean, t_mean, s_sdev, t_sdev]:
        print(repr(a[clas]))
    print('')

