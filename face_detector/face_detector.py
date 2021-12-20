import cv2
import os
import glob


IMAGES_FOLDER_PATH = r'images/'
RESIZED_IMAGES_FOLDER_PATH = r'resized_images/'
IMAGE_SIZE = 224


def resize_image(image, x, x2, y, y2):
	spread = 0
	cropped_image = image[y - spread : y2 + spread, x - spread : x2 + spread]
	dim = (IMAGE_SIZE, IMAGE_SIZE)

	return cv2.resize(cropped_image, dim, interpolation = cv2.INTER_AREA)


def save_resized_face_image(image, image_index, box_index, x, x2, y, y2):
	resized_image = resize_image(image, x, x2, y, y2)
	filename = str(image_index) + '_' + str(box_index) + '.jpg'
	os.chdir(RESIZED_IMAGES_FOLDER_PATH)
	cv2.imwrite(filename, resized_image)
	os.chdir('..')


def handle_image(image: any, image_index, classifier: any):
	boxes = classifier.detectMultiScale(image)
	box_index = 1

	for box in boxes:
		x, y, width, height = box
		x2, y2 = x + width, y + height
		save_resized_face_image(image, image_index, box_index, x, x2, y, y2)
		box_index += 1


def clear_resized_images():
	files = glob.glob(RESIZED_IMAGES_FOLDER_PATH + '*.jpg')
	for f in files:
         os.remove(f)


# ==========================================================================================================


clear_resized_images()

images_paths = list(filter(lambda k: 'jpg' in k, os.listdir(IMAGES_FOLDER_PATH)))
images_paths = [IMAGES_FOLDER_PATH + path for path in images_paths]

cascade_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

image_index = 1

for path in images_paths:
	handle_image(cv2.imread(path), image_index, cascade_classifier)
	image_index += 1
