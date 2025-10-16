import numpy as np
from scipy.signal import find_peaks


def window_smooth(points: np.ndarray, window_size: int = 10) -> np.ndarray:
    max_idx = len(points)
    half_size = window_size // 2
    smoothed_points = []
    for i, val in enumerate(points):
        l = max(0, i - half_size)
        r = min(max_idx, i + half_size)
        smoothed_points.append(points[l:r].mean())
    return np.array(smoothed_points)


def find_threshold(points: np.ndarray, distance: int = 30, prominence: float = 1e-06) -> int:
    peaks, props = find_peaks(points, distance=distance, prominence=prominence)
    assert len(peaks) >= 2, "Not enough peaks found"
    order = np.argsort(props['prominences'])
    top_peaks = peaks[order][-2:]
    midpoint = min(top_peaks) + abs(top_peaks[0] - top_peaks[1]) / 2
    return midpoint
