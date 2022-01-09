import cv2
import os
import glob

def horizontal_flip(image):
    return cv2.flip(image, 1)

def vertical_flip(image):
    return cv2.flip(image, 0)

def increase_brightness(image):
    mutation = 20
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    limit = 255 - mutation
    v[v > limit] = 255
    v[v <= limit] += mutation

    final_hsv = cv2.merge((h, s, v))
    final_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return final_image


