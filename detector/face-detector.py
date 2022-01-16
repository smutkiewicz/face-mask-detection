import cv2
import os
import glob
import pandas
import numpy

# switch between masked and no-masked images
# IMAGES_FOLDER_PATH = r'regular-images/'
IMAGES_FOLDER_PATH = r'masked-images/'
RESIZED_IMAGES_FOLDER_PATH = r'resized-images/'
MASKED_RESIZED_IMAGES_FOLDER_PATH = r'masked-resized-images/'
UNMASKED_RESIZED_IMAGES_FOLDER_PATH = r'unmasked-resized-images/'
IMAGE_SIZE = 224


# test function used to calibrate classifier
def show_image_with_indications():
	image = cv2.imread(IMAGES_FOLDER_PATH + 'test1.jpg')
	classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

	boxes = classifier.detectMultiScale(image, 1.2, 5)
	for box in boxes:
		x, y, width, height = box
		x2, y2 = x + width, y + height
		cv2.rectangle(image, (x, y), (x2, y2), (0,0,255), 1)

	cv2.imshow('TEST', image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def resize_image(image: any, x: int, x2: int, y: int, y2: int):
	spread = 10
	cropped_image = image[y - spread : y2 + spread, x - spread : x2 + spread]
	dim = (IMAGE_SIZE, IMAGE_SIZE)

	return cv2.resize(cropped_image, dim, interpolation = cv2.INTER_AREA)


def save_resized_face_image(image: any, image_index: int, box_index: int, x: int, x2: int, y: int, y2: int):
	resized_image = resize_image(image, x, x2, y, y2)
	filename = 'm-' + str(image_index) + '-' + str(box_index) + '.jpg'
	os.chdir(RESIZED_IMAGES_FOLDER_PATH)
	cv2.imwrite(filename, resized_image)
	os.chdir('..')
	os.chdir(MASKED_RESIZED_IMAGES_FOLDER_PATH)
	cv2.imwrite(filename, resized_image)
	os.chdir('..')


def handle_image(image: any, image_index: int, classifier: any):
	boxes = classifier.detectMultiScale(image, 1.2, 5)
	box_index = 1

	for box in boxes:
		x, y, width, height = box
		x2, y2 = x + width, y + height
		save_resized_face_image(image, image_index, box_index, x, x2, y, y2)
		box_index += 1


def clear_resized_images():
	files = glob.glob(RESIZED_IMAGES_FOLDER_PATH + '*.jpg')
	files.append
	for f in files:
         os.remove(f)


def generate_csv():
	csv_data = []

	images_paths = list(filter(lambda k: '.jpg' in k, os.listdir(MASKED_RESIZED_IMAGES_FOLDER_PATH)))

	for path in images_paths:
		csv_data.append([0, 0, 0, 0, 'with_mask', '1.jpg', 0, 0, '1', '1', path])

	images_paths = list(filter(lambda k: '.jpg' in k, os.listdir(UNMASKED_RESIZED_IMAGES_FOLDER_PATH)))

	for path in images_paths:
		csv_data.append([0, 0, 0, 0, 'without_mask', '1.jpg', 0, 0, '1', '1', path])

	print(csv_data)

	dataframe = pandas.DataFrame(numpy.array(csv_data), columns=['xmin', 'ymin', 'xmax', 'ymax', 'label', 'file', 'width', 'height', 'annotation_file', 'image_file', 'cropped_image_file'])
	dataframe.to_csv('images.csv', index=False)


def main() :
	# clear_resized_images()

	images_paths = list(filter(lambda k: 'jpg' in k, os.listdir(IMAGES_FOLDER_PATH)))
	images_paths = [IMAGES_FOLDER_PATH + path for path in images_paths]

	cascade_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

	image_index = 1

	for path in images_paths:
		handle_image(cv2.imread(path), image_index, cascade_classifier)
		image_index += 1

	generate_csv()
	# uncomment to test a single image
	# show_image_with_indications()




# ==========================================================================================================




main()
