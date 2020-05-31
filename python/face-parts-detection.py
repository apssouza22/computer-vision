import numpy as np
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2

args = {}
args["shape_predictor"] = "../downloads/models-cia/shape_predictor_68_face_landmarks.dat"

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 38

MOUTH_AR_CONSEC_FRAMES = 20
MOUTH_AR_THRESH = 0.90

# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
EYE_COUNTER = 0
MOUTH_COUNTER = 0


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

(mouthStart, mouthEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

def roi_aspect_ratio(roi):
	# compute the euclidean distances between the two sets of
	# vertical roi landmarks (x, y)-coordinates
	A = dist.euclidean(roi[1], roi[5])
	B = dist.euclidean(roi[2], roi[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(roi[0], roi[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear



def drowsiness_process(ratio):
	global EYE_COUNTER
	# check to see if the eye aspect ratio is below the blink
	# threshold, and if so, increment the blink frame counter
	if ratio < EYE_AR_THRESH:
		EYE_COUNTER += 1

		# if the eyes were closed for a sufficient number of
		# then sound the alarm
		if EYE_COUNTER >= EYE_AR_CONSEC_FRAMES:
			# draw an alarm on the frame
			cv2.putText(frame, "eyes closed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	# otherwise, the eye aspect ratio is not below the blink
	# threshold, so reset the counter and alarm
	else:
		EYE_COUNTER = 0


# loop over frames from the video stream
def happiness_process(ratio):
	global MOUTH_COUNTER
	# check to see if the mouth aspect ratio is below the threshold, and if so, increment the mount frame counter
	if ratio > MOUTH_AR_THRESH:
		MOUTH_COUNTER += 1

		# if the eyes were closed for a sufficient number of
		# then sound the alarm
		if MOUTH_COUNTER >= MOUTH_AR_CONSEC_FRAMES:
			# draw an alarm on the frame
			cv2.putText(frame, "Smilling!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

	# otherwise, the eye aspect ratio is not below the threshold, so reset the counter and alarm
	else:
		MOUTH_COUNTER = 0

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

while True:
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = roi_aspect_ratio(leftEye)
		rightEAR = roi_aspect_ratio(rightEye)

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# Doing the same process for the mouth
		mouth = shape[mouthStart:mouthEnd]
		mouthEAR = roi_aspect_ratio(mouth)
		mouthHull = cv2.convexHull(mouth)
		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

		drowsiness_process(ear)
		happiness_process(mouthEAR)

		# draw the computed eye aspect ratio on the frame to help
		# with debugging and setting the correct eye aspect ratio
		# thresholds and frame counters
		cv2.putText(frame, "EAR: {:.2f}".format(mouthEAR), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# loop over all face parts
		for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
			# loop over the subset of facial landmarks, drawing the
			# specific face part
			for (x, y) in shape[i:j]:
				cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

			# extract the ROI of the face region as a separate image
			(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
			roi = frame[y:y + h, x:x + w]
			if roi.shape[0] > 0:
				roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
