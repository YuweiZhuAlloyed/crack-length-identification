import math
from typing import Tuple
import numpy as np


def fit_line(xs: np.ndarray, ys: np.ndarray) -> Tuple[int, int]:
    n = xs.shape[0]
    x_mean = xs.mean()
    y_mean = ys.mean()
    a = ((xs*ys).sum() - n*x_mean*y_mean) / ((xs**2).sum() - n*x_mean**2)
    b = y_mean - a*x_mean
    # print(xs, ys)
    return a, b


def find_min_gradient(xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = xs.shape[0]
    min_grad = math.inf
    cxs, cys = None, None
    for i in range(n):
        for j in range(i+1, n):
            if xs[j] - xs[i] == 0:
                continue
            gradient = abs((ys[j] - ys[i]) / (xs[j] - xs[i]))
            if gradient < min_grad:
                min_grad = gradient
                cxs = np.array([xs[i], xs[j]])
                cys = np.array([ys[i], ys[j]])
    return cxs, cys
