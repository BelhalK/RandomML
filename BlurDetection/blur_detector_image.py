# python blur_detector_image.py --image path/to/image.png --thresh threshold

from utils.blur_detector import detect_blur_fft
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path input image that we'll detect blur in")
ap.add_argument("-t", "--thresh", type=int, default=20,
	help="threshold for our blur detector to fire")
ap.add_argument("-v", "--vis", type=int, default=-1,
	help="whether or not we are visualizing intermediary steps")
ap.add_argument("-d", "--test", type=int, default=-1,
	help="whether or not we should progressively blur the image")
args = vars(ap.parse_args())

# grayscale images
orig = cv2.imread(args["image"])
orig = imutils.resize(orig, width=500)
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

# Fourrier Transform based blur detection
(mean, blurry) = detect_blur_fft(gray, size=60,
	thresh=args["thresh"], vis=args["vis"] > 0)

# Display on image
image = np.dstack([gray] * 3)
color = (0, 0, 255) if blurry else (0, 255, 0)
text = "Blurry ({:.4f})" if blurry else "Not Blurry ({:.4f})"
text = text.format(mean)
cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
	color, 2)
print("[INFO] {}".format(text))

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
