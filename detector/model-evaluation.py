print('test')

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from pandas import DataFrame
from keras_preprocessing.image import ImageDataGenerator

from PIL import Image
from sklearn.metrics import accuracy_score

BATCH_SIZE = 32
key_cropped = 'cropped_image_file'
key_label = 'label'
cropped_dir = './resized-images'
test_df_file_name = './images.csv'
classifier_path = "./classifier.h5"

model = tf.keras.models.load_model(classifier_path)
# model = classifier

SEED_SIZE = 42

test_df = pd.read_csv(test_df_file_name)

image_target_size = (224, 224)

test_image_generator = ImageDataGenerator(rescale = 1. / 255.)
test_generator = test_image_generator.flow_from_dataframe(
    dataframe = test_df,
    directory = cropped_dir,
    x_col = key_cropped,
    y_col = key_label,
    batch_size = BATCH_SIZE,
    seed = SEED_SIZE,
    shuffle = True,
    class_mode = 'categorical',
    target_size = image_target_size,
    validate_filenames=False
)

generator = test_generator
generator.reset()

pred = model.predict(generator, 
                     batch_size=BATCH_SIZE,
                     steps=len(generator), 
                     verbose=1)




y_pred = np.argmax(pred, axis=1)


lin_score = accuracy_score(generator.labels, y_pred)
print('Score Report: ', lin_score)