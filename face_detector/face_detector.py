import cv2
import os


# IMAGES_FOLDER_PATH = r'face_detector/images/'
# RESIZED_IMAGES_FOLDER_PATH = r'face_detector/resized_images/'
IMAGES_FOLDER_PATH = r'images/'
RESIZED_IMAGES_FOLDER_PATH = r'resized_images/'
IMAGE_SIZE = 224


def resize_image(image, x, x2, y, y2):
	spread = 0
	cropped_image = image[y - spread : y2 + spread, x - spread : x2 + spread]
	dim = (IMAGE_SIZE, IMAGE_SIZE)

	# print(cropped_image.shape, 'shape', image.shape, '\n')

	resized = cv2.resize(cropped_image, dim, interpolation = cv2.INTER_AREA)

	return resized


def save_resized_face_image(image, classifier, image_index, box_index, x, x2, y, y2):
	resized_image = resize_image(image, x, x2, y, y2)
	os.chdir(RESIZED_IMAGES_FOLDER_PATH)
	cv2.imwrite(str(image_index) + '_' + str(box_index) + '.jpg', resized_image)
	os.chdir('..')


def handle_image(image: any, image_index, classifier: any):
	boxes = classifier.detectMultiScale(image)
	box_index = 1

	# x, y, width, height = boxes[0]
	# x2, y2 = x + width, y + height
	# draw a rectangle over the pixels
	#cv2.rectangle(image, (x, y), (x2, y2), (0,0,255), 1)
	# save_resized_face_image(image, classifier, image_index, box_index, x, x2, y, y2)

	for box in boxes:
		# extract
		x, y, width, height = box
		x2, y2 = x + width, y + height
		# draw a rectangle over the pixels
		#cv2.rectangle(image, (x, y), (x2, y2), (0,0,255), 1)
		save_resized_face_image(image, classifier, image_index, box_index, x, x2, y, y2)
		box_index += 1


	# # show the image
	# cv2.imshow('face detection', image)

	# # keep the window open until we press a key
	# cv2.waitKey(0)

	# # close the window
	# cv2.destroyAllWindows()

images_paths = list(filter(lambda k: 'jpg' in k, os.listdir(IMAGES_FOLDER_PATH)))
images_paths = [IMAGES_FOLDER_PATH + path for path in images_paths]

image = cv2.imread(images_paths[0])

# load the pre-trained model
cascade_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

image_index = 1

for path in images_paths:
	handle_image(cv2.imread(path), image_index, cascade_classifier)
	image_index += 1

# # perform face detection
# boxes = cascade_classifier.detectMultiScale(image)

# # print bounding box for each detected face
# for box in boxes:
# 	# extract
# 	x, y, width, height = box
# 	x2, y2 = x + width, y + height
# 	# draw a rectangle over the pixels
# 	cv2.rectangle(image, (x, y), (x2, y2), (0,0,255), 1)

