import numpy as np
import cv2

# This file is a set of commonly used functions by the viz scripts. It
# is not meant to be run on its own

def unblockshaped(arr, h, w):
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
		.swapaxes(1,2)
		.reshape(h, w))

def reshape_to_row(arr):
    grid = np.array([np.reshape(img, (28, 28)) for img in arr])
    return unblockshaped(grid, int(28), int(28 * grid.shape[0]))

def reshape_to_grid(arr):
    grid = np.array([np.reshape(img, (28, 28)) for img in arr])
    size = int(28 * np.sqrt(grid.shape[0]))
    return unblockshaped(grid, size, size)

