import cv2
import numpy as np

class ShapeDescriptor:

    def normalise(self,magnitudes):
        half_samples = (len(magnitudes) // 2) + 1
        return (magnitudes[1:half_samples] / magnitudes[0]).flatten()

    def convert1D(self,contour, no_samples, contour_centroids=None):
        if contour_centroids is None:
            contour_centroids = np.squeeze(contour.mean(axis=0))
        sample_points = np.squeeze(cv2.ximgproc.contourSampling(contour, no_samples))

        return np.sqrt(np.sum((sample_points - contour_centroids) ** 2, axis=1))

    def findLine(self,image, find_thresh=True):
        if find_thresh:
            _, thresh_image = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            thresh_image = image.copy()
        img_floodfill = thresh_image.copy()
        h, w = thresh_image.shape[:2]
        contour, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contour, key=cv2.contourArea)

        return contour

    def fourierDescriptor(self,contour, contour_centroids=None, no_samples=64):

        contour1d = self.convert1D(contour, contour_centroids=contour_centroids, no_samples=no_samples)
        fourier_transform = cv2.dft(contour1d, flags=cv2.DFT_COMPLEX_OUTPUT)
        magnitudes = cv2.magnitude(fourier_transform[:, :, 0], fourier_transform[:, :, 1])

        return self.normalise(magnitudes)

    def extractFeatures(self,image, find_thresh=True):
        contour = self.findLine(image, find_thresh=find_thresh)
        return self.fourierDescriptor(contour)