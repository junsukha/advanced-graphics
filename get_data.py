import os

# params_path = "car03/calib/optim_params.txt"
# images_path = 

# cam_list = []
# with os.scandir('car03/segmented_ngp') as entries:
    
#     for entry in entries:
#         with open(entry.name, 'r') as f:
#             data = f.read()

#         cam_list.append(entry.name)
    
#     cam_list = sorted(l, key=lambda x: x[3:5])
    

# with os.scandir('car03/calib') as entries:
#     for entry in entries:
#         print(entry.name)


import os
from PIL import Image
import cv2
from typing import Tuple
def get_images_data(images_path='car03/segmented_ngp'):
    '''
    images_path should be 'car03/segmented_ngp
    Returns (images, cam2world)
    '''
    # Path to the directory containing folders with images
    # images_path = 'car03/segmented_ngp'

    images = []
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
    for file_path in file_paths:
        # img = cv2.imread(file_path)
        # print(type(img))

        image = Image.open(file_path)
        image.thumbnail((128, 128))
        # image.save('image_thumbnail.png')
        # print(image.size) # Output: (400, 350)
        # exit(0)
        # print(file_path)
        # print(type(image))
        # exit(0)
        
        # image = np.array(image)
        # print(image[:3,:3,:])
        # exit(1)
        # mask = (image == 0.0)
        # mask[..., 2] = 1.0
        # print(f'mask shape: {mask.shape}')
        # image[mask == 0.0] = 1.0 
        image = np.array(image).astype(float) / 255
        # print(image.shape)
        
        for i in range (image.shape[0]):
            for j in range(image.shape[1]):
                # print(image[i,j,:3])
                # exit(0)
                if image[i,j,0] == 0 and image[i,j,1] == 0 and image[i,j,2] == 0 :
                    image[i,j,:3] = [1.0,1,1]
        # image[..., :3]
        # exit(0)
        images.append(image.tolist())
        # cv2.imwrite('test.png',image)
        # exit(0)
    # print(f'images.shape: {images[0].shape}') # 720, 1280, 3
    # print(f'images.shape: {np.array(images).shape}') # 53, 720, 1280, 3
    return np.array(images)

from scipy.spatial.transform import Rotation as R
import numpy as np

def get_cam2world_data(path='car03/calib/optim_params.txt'):
    # data_dict = {}
    rot_mats = []
    cam2worlds = []
    ixts = []
    with open(path, "r") as f:
        next(f)
        for line in f:
        # data = f.readline().split(' ')
            data = line.rstrip('\n').split(' ')    
            t = np.array(list(map(float, [data[-3], data[-2], data[-1]])))
            rot_mat = R.from_quat(list(map(float,[data[-6], data[-5], data[-4], data[-7]])))
            rot_mat = np.array(rot_mat.as_matrix())

            cam2world = np.concatenate((rot_mat, t.reshape(-1,1)), axis=-1)
            cam2world = np.vstack((cam2world, np.array([0,0,0,1])))
            
            cam2worlds.append(cam2world.tolist())
          
            ixt = get_ixt(data)   
                  
            ixts.append(ixt)

    # print(len(rot_mats))
    # print(rot_mats[0])

    cam2worlds = np.array(cam2worlds)
    
    ixts = np.array(ixts)
 
    # print(type(cam2worlds))
    # print(cam2worlds[:2])
    # print(cam2worlds.shape)
    return cam2worlds, ixts
def get_ixt(data):
   
    fx = float(data[3])
    fy = float(data[4])
    cx = float(data[5])
    cy = float(data[6])
    ixt = [[fx,  0.0, cx],
           [ 0.0, fy, cy],
           [ 0.0,  0.0,  1.0]]
    return ixt


if __name__ == "__main__":
    get_images_data('car03/segmented_ngp')
    # get_cam2world_data()