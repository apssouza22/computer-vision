import matplotlib.pyplot as plt
import cv2


def get_center(image):
	# grab the dimensions and compute the center of the image
	(h, w) = image.shape[:2]
	(cX, cY) = (int(w * 0.5), int(h * 0.5))
	return cX, cY


def display(img, cmap='gray', bgr2rgb=True):
	# Defining a util function to display image using matplotlib
	flg = plt.figure(figsize=(12, 10))
	ax = flg.add_subplot(111)
	if bgr2rgb:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	ax.imshow(img, cmap)
