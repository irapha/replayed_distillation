import numpy as np
import datasets as d
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from procedures.replay import reshape_to_row

mnist = d.get('mnist')
recns = np.load('data_optimized_notmedian_relumse.npy')[()]
#  recns_notmedian = np.load('data_optimized_notmedian.npy')[()]

og_means = []
for clas in range(10):
    idx = np.where(np.where(mnist.og.train.labels == 1)[1] == clas)[0]
    og_means.append(np.mean(mnist.og.train.images[idx], axis=0))
cv2.imshow('og means', reshape_to_row(np.array(og_means)))

re_means = []
for clas in range(10):
    clas_imgs = []
    for batch_x, _ in recns[clas]:
        clas_imgs.extend(batch_x)
    re_means.append(np.mean(clas_imgs, axis=0))
cv2.imshow('re means notmedian centralnorm relumse', reshape_to_row(np.array(re_means)))

#  cv2.waitKey(0)
#  cv2.destroyAllWindows()


# analysing brightness and contrast
f, subs = plt.subplots(1, 10, sharey=True, sharex=True)
x = np.linspace(-1.0, 2.0, 30)
plt.locator_params(axis='y', nbins=3)
plt.locator_params(axis='x', nbins=2)

for i, (og, re) in enumerate(zip(og_means, re_means)):
    subs[i].hist(og, x, facecolor='red', alpha=0.5)

    subs[i].hist(re, x, facecolor='blue', alpha=0.5)

    subs[i].plot(x, mlab.normpdf(x, np.mean(og), np.sqrt(np.var(og))), 'r')
    subs[i].plot(x, mlab.normpdf(x, np.mean(re), np.sqrt(np.var(re))), 'b')

print('og is red, re is blue')
plt.show()
