import numpy as np
import cv2

# This file is a set of commonly used functions by the viz scripts. It
# is not meant to be run on its own

def unblockshaped(arr, h, w, rgb=False):
    if rgb:
        n, nrows, ncols, nchannels = arr.shape
        return (arr.reshape(h//nrows, -1, nrows, ncols, nchannels)
                    .swapaxes(1,2)
                    .reshape(h, w, 3))
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
		.swapaxes(1,2)
		.reshape(h, w))

def reshape_to_row(arr, side=28, rgb=False):
    if rgb:
        grid = np.array([np.reshape(img, (side, side, 3)) for img in arr])
    else:
        grid = np.array([np.reshape(img, (side, side)) for img in arr])
    return unblockshaped(grid, int(side), int(side * grid.shape[0]), rgb=rgb)

def reshape_to_grid(arr, side=28, rgb=False):
    if rgb:
        grid = np.array([np.reshape(img, (side, side, 3)) for img in arr])
    else:
        grid = np.array([np.reshape(img, (side, side)) for img in arr])
    size = int(side * np.sqrt(grid.shape[0]))
    return unblockshaped(grid, size, size, rgb=rgb)

