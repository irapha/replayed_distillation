import numpy as np
import datasets as d
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from procedures.replay import reshape_to_row
from random import choice

mnist = d.get('mnist')
recns = np.load('stats/data_optimized_replay_drop_rescaleonly.npy')[()]
#  recns_notmedian = np.load('data_optimized_notmedian.npy')[()]

og_means = []
for clas in range(10):
    idx = np.where(np.where(mnist.og.train.labels == 1)[1] == clas)[0]
    og_means.append(np.mean(mnist.og.train.images[idx], axis=0))
#  cv2.imwrite('og_means.png', 255* reshape_to_row(np.array(og_means)))
cv2.imshow('og_means.png', reshape_to_row(np.array(og_means)))

re_means = []
for clas in range(10):
    clas_imgs = []
    for batch_x, _ in recns[clas]:
        clas_imgs.extend(batch_x)
    re_means.append(np.mean(clas_imgs, axis=0))
#  cv2.imwrite('re_means.png', 255*reshape_to_row(np.array(re_means)))
cv2.imshow('re_means.png', reshape_to_row(np.array(re_means)))

re_random = []
for clas in range(10):
    clas_imgs = []
    for batch_x, _ in recns[clas]:
        clas_imgs.extend(batch_x)
    re_random.append(choice(clas_imgs))
#  cv2.imwrite('re_random.png', 255*reshape_to_row(np.array(re_random)))
cv2.imshow('re_random.png', reshape_to_row(np.array(re_random)))

cv2.waitKey(0)
cv2.destroyAllWindows()


# analysing brightness and contrast
f, subs = plt.subplots(1, 10, sharey=True, sharex=True, figsize=(40, 3))
x = np.linspace(0.0, 1.0, 30)
plt.locator_params(axis='y', nbins=3)
plt.locator_params(axis='x', nbins=2)

for i, (og, re) in enumerate(zip(og_means, re_means)):

    grey = '#666666'
    light_grey = '#b7b7b7'
    subs[i].tick_params(axis='x', colors=grey)
    #  ax.xaxis.label.set_color(grey)
    subs[i].tick_params(axis='y', colors=grey)
    subs[i].spines['bottom'].set_color(grey)
    subs[i].spines['top'].set_color(grey)
    subs[i].spines['left'].set_color(grey)
    subs[i].spines['right'].set_color(grey)
    #  ax.yaxis.label.set_color(grey)

    subs[i].hist(og, x, facecolor='#a64d79', alpha=0.5)
    subs[i].hist(re, x, facecolor='#674ea7', alpha=0.5)

    #  subs[i].plot(x, mlab.normpdf(x, np.mean(og), np.sqrt(np.var(og))), 'r')
    #  subs[i].plot(x, mlab.normpdf(x, np.mean(re), np.sqrt(np.var(re))), 'b')

print('og is red, re is blue')
#  plt.show()
plt.savefig('data_reconstruction.png', dpi=300)
