import numpy as np
import datasets as d
import cv2
from procedures.replay import reshape_to_row

mnist = d.get('mnist')
recns = np.load('data_optimized_notmedian_centralnorm.npy')[()]
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
cv2.imshow('re means notmedian centralnorm', reshape_to_row(np.array(re_means)))

#  re_means = []
#  for clas in range(10):
    #  clas_imgs = []
    #  for batch_x, _ in recns_notmedian[clas]:
        #  clas_imgs.extend(batch_x)
    #  re_means.append(np.mean(clas_imgs, axis=0))
#  cv2.imshow('re means notmedian', reshape_to_row(np.array(re_means)))

cv2.waitKey(0)
cv2.destroyAllWindows()
