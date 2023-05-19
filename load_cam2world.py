import numpy as np

cam2world = np.load('./cam2world.npy')

# print(cam2world.shape)
# print(cam2world[0]) # [[ 1.  0.  0. -0.]
                    #  [ 0. -1.  0. -0.]
                    #  [ 0.  0. -1.  3.]
                    #  [ 0.  0.  0.  1.]]

for c2w in cam2world:
    print(c2w)
    break


import json
 
# Data to be written
# dictionary = {
#     "name": "sathiyajith",
#     "rollno": 56,
#     "cgpa": 8.6,
#     "phonenumber": "9976770500"
# }


# intrinsics = torch.tensor([[0.7, 0., 0.5], # Copied from last assignment
#                            [0., 0.7, 0.5],
#                            [0., 0., 1.]]).to(images.device)

# (128, 128, 3)

dictionary = {
    "camera_angle_x": 0.5,
    "camera_angle_y": 0.5,
    "fl_x": 0.7,
    "fl_y": 0.7,
    "k1": 0,
    "k2": 0,
    "k3": 0,
    "k4": 0,
    "p1": 0,
    "p2": 0,
    "is_fisheye": False,
    "cx": 0.5,
    "cy": 0.5,
    "w": 128,
    "h": 128,
    "aabb_scale": 16,
    "frames": []
}


for i, c2w in enumerate(cam2world):
    dict = {}
    dict["file_path"] = "./data/image{:03d}.jpg".format(i)
    dict["sharpness"] = 50
    dict["transform_matrix"] = c2w.tolist()
    dictionary["frames"].append(dict)


# for 
 
# Serializing json
json_object = json.dumps(dictionary, indent=2)
 
# Writing to sample.json
with open("sample.json", "w") as outfile:
    outfile.write(json_object)

