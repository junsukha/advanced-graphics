import os
from PIL import Image
import cv2
from typing import Tuple
import numpy as np

def save_images(images_path='car03/segmented_ngp'):
    '''
    images_path should be 'car03/segmented_ngp
    Returns (images, cam2world)
    '''
    # Path to the directory containing folders with images
    # images_path = 'car03/segmented_ngp'


    file_paths = []
    # Loop through each folder in the directory
    for folder_name in os.listdir(images_path):
        image_folder_path = os.path.join(images_path, folder_name)
        
        # Loop through each file in the folder
        for file_name in os.listdir(image_folder_path):
            file_path = os.path.join(image_folder_path, file_name)
            file_paths.append(file_path)
            
            # Check if the file is an image
            # if file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png'):
            
    # sort file paths
    file_paths = sorted(file_paths, key=lambda x: x[-15:-13])         

    # read images
    for i, file_path in enumerate(file_paths):
        img = cv2.imread(file_path)
    
        cv2.imwrite('car/image{:03d}.png'.format(i),img)
 

if __name__ == "__main__":
    save_images()