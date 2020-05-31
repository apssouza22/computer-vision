import numpy as np
import cv2
import imutils

from pyimagesearch.imageutils import get_center


class ColorDescriptor:
	"""
	This class will encapsulate all the necessary logic to extract our 3D HSV color histogram from our images
	"""

	def __init__(self, bins):
		"""
		Store the number of bins for the 3D histogram
		For our photo image search engine, we’ll be utilizing a 3D color histogram in the HSV color space
		with 8 bins for the Hue channel, 12 bins for the saturation channel, and 3 bins for the value channel,
		yielding a total feature vector of dimension 8 x 12 x 3 = 288.

		:param bins:
		"""
		self.bins = bins

	def describe(self, image):
		# convert the image to the HSV color space and initialize
		# the features used to quantify the image
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		features = []

		# grab the dimensions and compute the center of the image
		(h, w) = image.shape[:2]
		(cX, cY) = get_center(image)

		# divide the image into four rectangles/segments (top-left,
		# top-right, bottom-right, bottom-left)
		top_left = (0, cX, 0, cY)
		top_right = (cX, w, 0, cY)
		bottom_right = (cX, w, cY, h)
		bottom_left = (0, cX, cY, h)
		segments = [top_left, top_right, bottom_right, bottom_left]

		# construct an elliptical mask representing the center of the
		# image
		(axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
		ellipMask = np.zeros(image.shape[:2], dtype="uint8")
		cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

		# loop over the segments
		for (startX, endX, startY, endY) in segments:
			# construct a mask for each corner of the image, subtracting
			# the elliptical center from it
			cornerMask = np.zeros(image.shape[:2], dtype="uint8")
			cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
			cornerMask = cv2.subtract(cornerMask, ellipMask)

			# extract a color histogram from the image, then update the
			# feature vector
			hist = self.histogram(image, cornerMask)
			features.extend(hist)

		# extract a color histogram from the elliptical region and
		# update the feature vector
		hist = self.histogram(image, ellipMask)
		features.extend(hist)

		# return the feature vector
		return features

	def histogram(self, image, mask):
		# extract a 3D color histogram from the masked region of the
		# image, using the supplied number of bins per channel
		hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 180, 0, 256, 0, 256])
		hist = cv2.normalize(hist, hist).flatten()
		# return the histogram
		return hist
