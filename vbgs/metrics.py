import numpy as np


def calc_mse(x, y):
    if x.max() <= 1:
        x = x * 255
    if y.max() <= 1:
        y = y * 255

    return np.mean((1.0 * x - 1.0 * y) ** 2)


def calc_psnr(x, y):
    if x.max() <= 1:
        x = x * 255
    if y.max() <= 1:
        y = y * 255

    mse_ = calc_mse(x, y)
    return 20 * np.log10(254 / np.sqrt(mse_))
