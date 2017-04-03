import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

all_stats = np.load('all_stats.npy')[()]

s_mean, s_sdev = all_stats['student_stats']
t_mean, t_sdev = all_stats['teacher_stats']

f, subs = plt.subplots(10, 10, sharey=True, sharex=True)
x = np.linspace(-1400, 1400, 100)

plt.locator_params(axis='y', nbins=3)
plt.locator_params(axis='x', nbins=2)

for clas in range(10):
    for att in range(10):
        subs[clas, att].plot(x, mlab.normpdf(x, s_mean[clas][att], s_sdev[clas][att]), 'r')
        subs[clas, att].plot(x, mlab.normpdf(x, t_mean[clas][att], t_sdev[clas][att]), 'b')
        # subs[clas, att].set_title('class: {}, attr: {}'.format(clas, att))

print('teacher is blue, student is red')
plt.setp([a.get_xticklabels() for a in subs[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in subs[:, 1]], visible=False)
plt.show()

