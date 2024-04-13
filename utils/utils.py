import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

def read_data(directory, target_size = (300, 300)):
    all_files = os.listdir(directory)
    not_mask_images = [file for file in all_files if "mask" not in file]
    mask_images = [file for file in all_files if "mask" in file]
    images = []
    masks = []
    temp_masks = []

    for file_name in not_mask_images:
        image = cv2.imread(os.path.join(directory, file_name))
        images.append(cv2.resize(image, target_size))
    
    prev_file = None
    for file_name in mask_images:
        image = cv2.imread(os.path.join(directory, file_name), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, target_size)
        if prev_file != None and (str(prev_file).replace(".png", "").replace(" ", "_") in str(file_name).replace(".png", "").replace(" ","_")):
            masks[-1] += image
        else:
            temp_masks.append(file_name)
            masks.append(image)
            prev_file = file_name

    return np.array(images), np.array(masks)

def show_image(data, index):
    image = data[index]
    plt.imshow(image)
    plt.axis('off')
    plt.show()