import cv2
import os
import glob


TEST_IMAGES_FOLDER_PATH = r'test-images/'
MODIFIED_IMAGES_FOLDER_PATH = r'modified-images/'


def horizontal_flip(image: any):
    return cv2.flip(image, 1)


def vertical_flip(image: any):
    return cv2.flip(image, 0)


def increase_brightness(image: any):
    mutation = 70
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    limit = 255 - mutation
    v[v > limit] = 255
    v[v <= limit] += mutation

    final_hsv = cv2.merge((h, s, v))
    final_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return final_image


def clear_modified_images():
	files = glob.glob(MODIFIED_IMAGES_FOLDER_PATH + '*.jpg')
	for f in files:
         os.remove(f)


def save_image(image: any, filename: str):
    os.chdir(MODIFIED_IMAGES_FOLDER_PATH)
    cv2.imwrite(filename + ".jpg", image)
    os.chdir('..')


def modify_and_save_images(image: any, image_index: int, brighten = False):
    image[brighten] = increase_brightness(image)
    suffix = "-b" if brighten else "-"

    save_image(image, str(image_index) + suffix)
    save_image(horizontal_flip(image), str(image_index) + suffix + "h")
    save_image(vertical_flip(image), str(image_index) + suffix + "v")
    save_image(horizontal_flip(vertical_flip(image)), str(image_index) + suffix + "hv")


def handle_image(image: any, image_index: int):
    modify_and_save_images(image, image_index)
    modify_and_save_images(image, image_index, True)


def main():
    clear_modified_images()

    images_paths = list(filter(lambda k: 'jpg' in k, os.listdir(TEST_IMAGES_FOLDER_PATH)))
    images_paths = [TEST_IMAGES_FOLDER_PATH + path for path in images_paths]

    image_index = 1

    for path in images_paths:
        handle_image(cv2.imread(path), image_index)
        image_index += 1




# ==========================================================================================================




main()