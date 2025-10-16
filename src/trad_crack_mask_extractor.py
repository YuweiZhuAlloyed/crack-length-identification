import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from src.interfaces import ICrackMaskExtractor
from src.processing import *


class ThresholdCrackMaskExtractor(ICrackMaskExtractor):

    def __init__(self, image_path: str, save_path: str):
        self.image_path = image_path
        self.save_path = save_path
        self.image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        # cv2.equalizeHist(self.image, self.image)

    def extract(self):
        heights, xs = np.histogram(
            self.image.flatten(), bins=255, density=True)
        smoothed_heights = window_smooth(heights, window_size=40)
        threshold = find_threshold(smoothed_heights)
        mask: np.ndarray = (self.image < threshold).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        output = np.zeros_like(mask)
        cv2.drawContours(output, [largest_contour], -
                         1, 255, thickness=cv2.FILLED)
        cv2.imwrite(self.save_path, output)


class GMMCrackMaskExtractor(ICrackMaskExtractor):

    def __init__(self, image_path: str, save_path: str):
        self.image_path = image_path
        self.save_path = save_path
        self.image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        # cv2.equalizeHist(self.image, self.image)

    def extract(self):

        sample = np.random.choice(
            self.image.flatten(), size=10000, replace=False)
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(sample.reshape(-1, 1))
        centroids = gmm.means_
        threshold = centroids.min() + (centroids.max() - centroids.min()) / 2
        mask: np.ndarray = (self.image < threshold).astype(np.uint8)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        output = np.zeros_like(mask)
        cv2.drawContours(output, [largest_contour], -
                         1, 255, thickness=cv2.FILLED)
        cv2.imwrite(self.save_path, output)
