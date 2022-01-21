import cv2
import os
import glob
import pandas
import copy
import numpy


TEST_IMAGES_FOLDER_PATH = r'test-images/'
MODIFIED_IMAGES_FOLDER_PATH = r'modified-images/'
IMAGES_TO_MODIFY_FOLDER_PATH = r'images-to-modify/'
CSV_TO_READ = 'train_df.csv'
CLASS_TO_AUGMENT = 'without_mask'


def horizontal_flip(image: any):
    return cv2.flip(image, 1)


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
	files = glob.glob(MODIFIED_IMAGES_FOLDER_PATH + '*.png')
	for f in files:
         os.remove(f)


def save_image(image: any, filename: str):
    os.chdir(MODIFIED_IMAGES_FOLDER_PATH)
    cv2.imwrite(filename + ".png", image)
    os.chdir('..')


def modify_and_save_images(image: any, file_suffix: str):
    suffix = "-"

    save_image(horizontal_flip(image), file_suffix[:-4] + suffix + "h")
    save_image(increase_brightness(image), file_suffix[:-4] + suffix + "b")
    save_image(increase_brightness(horizontal_flip(image)), file_suffix[:-4] + suffix + "bh")


def handle_image(image: any, file_suffix: str):
    modify_and_save_images(image, file_suffix)


def main():
    clear_modified_images()

    images_paths = list(filter(lambda k: 'jpg' in k, os.listdir(TEST_IMAGES_FOLDER_PATH)))
    images_paths = [TEST_IMAGES_FOLDER_PATH + path for path in images_paths]

    image_index = 1

    for path in images_paths:
        handle_image(cv2.imread(path), str(image_index))
        image_index += 1



# ==========================================================================================================



# method used to augment real-used dataset with the help of `train_df.csv`
def augment_data():
    clear_modified_images()
    csv = pandas.read_csv(CSV_TO_READ).to_numpy()

    mask_off = []

    #columns - xmin,ymin,xmax,ymax,label,file,width,height,annotation_file,image_file,cropped_image_file

    for element in csv:
     if element[4] == CLASS_TO_AUGMENT:
        mask_off.append(True)
     else:
        mask_off.append(False)

    csv = csv[mask_off]

    augmented_csv = []
    suffixes = ['-h', '-b', '-bh']
    for element in csv:
        handle_image(cv2.imread(IMAGES_TO_MODIFY_FOLDER_PATH + element[10]), element[10])

        for suffix in suffixes:
            position = copy.deepcopy(element)
            position[10] = position[10][:-4] + suffix + position[10][-4:]
            augmented_csv.append(position)

    dataframe = pandas.DataFrame(numpy.array(augmented_csv), columns=['xmin', 'ymin', 'xmax', 'ymax', 'label', 'file', 'width', 'height', 'annotation_file', 'image_file', 'cropped_image_file'])
    dataframe.to_csv('augmented_cropped_images.csv', index=False)




# ==========================================================================================================




main()
# augment_data()
