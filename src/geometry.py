from typing import Tuple
import numpy as np


def fit_line(xs: np.ndarray, ys: np.ndarray) -> Tuple[int, int]:
    n = xs.shape[0]
    x_mean = xs.mean()
    y_mean = ys.mean()
    a = ((xs*ys).sum() - n*x_mean*y_mean) / ((xs**2).sum() - n*x_mean**2)
    b = y_mean - a*x_mean
    return a, b
