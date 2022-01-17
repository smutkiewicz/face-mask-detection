import cv2
import os
import glob
import numpy
import tensorflow


IMAGES_TO_EXAMINE_FOLDER_PATH = r'images-to-examine/'
MARKED_IMAGES_FOLDER_PATH = r'marked-images/'
CLASSIFIER_PATH = "./classifier-simplified.h5"
IMAGE_SIZE = (24, 22)


# test function used to calibrate classifier
def show_image_with_indications():
	image = cv2.imread(MARKED_IMAGES_FOLDER_PATH + 'test1.jpg')
	classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

	boxes = classifier.detectMultiScale(image, 1.2, 5)
	for box in boxes:
		x, y, width, height = box
		x2, y2 = x + width, y + height
		cv2.rectangle(image, (x, y), (x2, y2), (0,0,255), 1)

	cv2.imshow('TEST', image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def clear_marked_images_folder():
	files = glob.glob(MARKED_IMAGES_FOLDER_PATH + '*.jpg')
	files.append
	for f in files:
         os.remove(f)


def resize_image(image: any, x: int, x2: int, y: int, y2: int):
	spread = 0
	cropped_image = image[y - spread : y2 + spread, x - spread : x2 + spread]

	return cv2.resize(cropped_image, IMAGE_SIZE, interpolation = cv2.INTER_AREA).reshape(1, 24, 22, 3)


def examine_image(image_name: str, image_path: str, model):
	image = cv2.imread(image_path)

	cascade_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

	# get boxes containing faces
	boxes = cascade_classifier.detectMultiScale(image, 1.2, 5)

	for box in boxes:
		x, y, width, height = box
		x2, y2 = x + width, y + height

		# get image fragment containing face in appropriate shape
		face_fragment = resize_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), x, x2, y, y2)

		# 0 - with, 1 - without
		pred = model.predict(face_fragment, verbose=0)

		color = (0,255,0)
		if numpy.argmax(pred, axis=1)[0] == 1:
			color = (0,0,255)

		cv2.rectangle(image, (x, y), (x2, y2), color, 5)

	# copy image to marked-images folde
	os.chdir(MARKED_IMAGES_FOLDER_PATH)
	cv2.imwrite('marked-' + image_name, image)
	os.chdir('..')


def main():
	clear_marked_images_folder()

	# get images to examine
	images_names = list(filter(lambda k: 'jpg' in k, os.listdir(IMAGES_TO_EXAMINE_FOLDER_PATH)))
	images_paths = [IMAGES_TO_EXAMINE_FOLDER_PATH + path for path in images_names]
	
	# load classifier
	model = tensorflow.keras.models.load_model(CLASSIFIER_PATH)

	for name, path in zip(images_names, images_paths):
		examine_image(name, path, model)




# ==========================================================================================================




main()
