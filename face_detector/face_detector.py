import cv2


# TODO temporary path which works, will try to change to relative path
image = cv2.imread(r"/Users/miazga/Documents/Studies/EIASR/face-mask-detection/face_detector/images/test3.jpg")

# load the pre-trained model
cascade_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# perform face detection
boxes = cascade_classifier.detectMultiScale(image)

# print bounding box for each detected face
for box in boxes:
	# extract
	x, y, width, height = box
	x2, y2 = x + width, y + height
	# draw a rectangle over the pixels
	cv2.rectangle(image, (x, y), (x2, y2), (0,0,255), 1)

# show the image
cv2.imshow('face detection', image)

# keep the window open until we press a key
cv2.waitKey(0)

# close the window
cv2.destroyAllWindows()