import time

import cv2
import numpy as np
from imutils.video import VideoStream

from sklearn.metrics import pairwise

from pyimagesearch.singlemotiondetector import SingleMotionDetector

background = None
accumulated_weight = 0.5

# Area of interest box
roi_top = 20
roi_bottom = 300
roi_right = 20
roi_left = 300

num_frames = 0
cont_color = (255, 0, 0)
conv_color = (0, 0, 255)
motion_detector = SingleMotionDetector(accumWeight=accumulated_weight)


def count_fingers(thresholded, conv_hull, frame_copy):
	# Now the convex hull will have at least 4 most outward points, on the top, bottom, left, and right.
	# Let's grab those points by using argmin and argmax. Keep in mind, this would require reading the documentation
	# And understanding the general array shape returned by the conv hull.

	# Find the top, bottom, left , and right.
	# Then make sure they are in tuple format
	top = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
	bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
	left = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
	right = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])

	cX = (left[0] + right[0]) // 2
	cY = (top[1] + bottom[1]) // 2

	distance = pairwise.euclidean_distances([[cX, cY]], Y=[left, right, top, bottom])[0]

	max_distance = distance.max()

	radius = int(0.8 * max_distance)
	circumfrence = (2 * np.pi * radius)
	circular_roi = np.zeros((thresholded.shape[0], thresholded.shape[1], 1), np.uint8)

	cv2.circle(circular_roi, (cX, cY), radius, 255, 10)

	# adding circle to the frame for debugging propose
	cv2.circle(frame_copy, (cX + roi_right, cY + roi_top), radius, 255, 10)

	circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

	contours, hierarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	count = 0

	for cnt in contours:

		(x, y, w, h) = cv2.boundingRect(cnt)

		# 1. Contour region is not the very bottom of hand area (the wrist)
		out_of_wrist = ((cY + (cY * 0.25)) > (y + h))

		# 2. Number of points along the contour does not exceed 25% of the circumference of the circular ROI (otherwise we're counting points off the hand)
		limit_points = ((circumfrence * 0.25) > cnt.shape[0])

		if out_of_wrist and limit_points:
			count += 1
	if count > 5:
		return 5
	return count


if __name__ == "__main__":
	cam = VideoStream(src=0).start()
	time.sleep(2.0)

	while True:

		frame = cam.read()
		frame_copy = frame.copy()
		roi = frame[roi_top:roi_bottom, roi_right:roi_left]
		gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (7, 7), 0)

		if num_frames <= 59:
			motion_detector.update(gray)
			cv2.putText(frame_copy, 'WAIT. GETTING BACKGROUND', (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
			cv2.imshow('Finger Count', frame_copy)

		if num_frames > 60:
			# detect motion in the image
			motion = motion_detector.detect(gray)

			if motion is not None:
				thresholded, hand_contour = motion

				# Getting the Boundary from a set of points(fingers)
				conv_hull = cv2.convexHull(hand_contour)
				cv2.drawContours(frame_copy, [hand_contour + (roi_right, roi_top)], -1, cont_color, 5)
				cv2.drawContours(frame_copy, [conv_hull + (roi_right, roi_top)], -1, conv_color, 1, 8)

				fingers = count_fingers(thresholded, conv_hull, frame_copy)
				cv2.drawContours(frame_copy, [conv_hull + (roi_right, roi_top)], -1, conv_color, 1, 8)

				cv2.putText(frame_copy, str(fingers), (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

				cv2.imshow('Thresholded', thresholded)

		cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0, 0, 255), 5)

		num_frames += 1

		cv2.imshow('Finger Count', frame_copy)

		k = cv2.waitKey(1) & 0xFF

		if k == 27:
			break

	cam.release()
	cv2.destroyAllWindows()
