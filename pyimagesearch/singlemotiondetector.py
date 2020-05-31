import numpy as np
import imutils
import cv2


class SingleMotionDetector:
	"""
	SingleMotionDetector detects movements in a video
	"""

	def __init__(self, accumWeight=0.5):
		# store the accumulated weight factor
		self.accumWeight = accumWeight

		# initialize the background model
		self.bg = None

	def update(self, image):
		# if the background model is None, initialize it
		if self.bg is None:
			self.bg = image.copy().astype("float")
			return

		# update the background model by accumulating the weighted
		# average
		cv2.accumulateWeighted(image, self.bg, self.accumWeight)

	def detect(self, image, tVal=25):
		# compute the absolute difference between the background model
		# and the image passed in, then threshold the delta image
		delta = cv2.absdiff(self.bg.astype("uint8"), image)
		thresh = cv2.threshold(delta, tVal, 255, cv2.THRESH_BINARY)[1]

		# perform a series of erosions and dilations to remove small
		# blobs
		thresh = cv2.erode(thresh, None, iterations=2)
		thresh = cv2.dilate(thresh, None, iterations=2)

		# find contours in the thresholded image and initialize the
		# minimum and maximum bounding box regions for motion
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)

		# if no contours were found, return None
		if len(cnts) == 0:
			return None
		else:
			# ASSUMING THE LARGEST EXTERNAL CONTOUR IN ROI, IS THE HAND
			hand_segment = max(cnts, key=cv2.contourArea)

			return thresh, hand_segment
