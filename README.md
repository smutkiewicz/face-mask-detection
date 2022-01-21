# face-mask-detection

Running face detector script:
1. Put the JPG files you want to examine into the `detector/images-to-examine` folder
2. Put the trained classifier file into the `detector` and make sure its name is appropriate for the `CLASSIFIER_PATH` constant in `face-detector.py` (line 10)
3. In the command line, navigate to the `detector` directory
4. Execute command `python3 face-detector.py`
5. Examined images will appear in `detector/marked-images` 


Running data augmentation (test mode):
1. Put the JPG files you want to modify into the `data-augmentation/test-images` folder
2. In the command line, navigate to the `data-augmentation` directory
3. Execute command `python3 data-augmentation.py`
4. Modified images will appear in `data-augmentation/modified-images` 


Running data augmentation (real application mode):
1. Put the files in a directory (name is to choose) and put this folder into `data-augmentation`
2. Provide the CSV file describing the data in `data-augmentation` folder
3. In `data-augmentation.py` change:
    3.1. `IMAGES_TO_MODIFY_FOLDER_PATH` (line 11) to the name of your folder
    3.2. `CSV_TO_READ` (line 12) to the name of your CSV file
    3.3. `CLASS_TO_AUGMENT` (line 13) to the value of class property you want to augment
    3.4. In line 87, set the proper index of the column where the class label is specified
    3.5. Uncomment line 116 and comment 115
4. In the command line, navigate to the `data-augmentation` directory
5. Execute command `python3 data-augmentation.py`
6. Modified images will appear in `data-augmentation/modified-images` along with generated CSV file in `data-augmentation`
