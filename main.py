
import os, sys
# from google.colab import drive
import matplotlib.pyplot as plt
import sys
import cv2


import pytorch3d
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import torch
import gc
print(f"Installed Torch version: {torch.__version__}")

# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
#     torch.cuda.set_device(device)
# else:
#     device = torch.device("cpu")
torch.cuda.empty_cache()

import numpy as np

# scikit-image lets us do basic image image-processing tasks.
import skimage
import skimage.transform

from mpl_toolkits.axes_grid1 import make_axes_locatable

from hybrid import *
from rest import *
from utils import *
from rendering import *

''' test hybrid voxel neural field'''
# hybrid = HybridVoxelNeuralField((16, 16, 16), feature_dim=8, out_dim=2).cuda() # 16,16,16 is grid's resolution (feature cube)

# # %%script echo skipping
# dataset = sphere_generator_random((32,32,32)) # this shape will be coord's shape. So Cube size is 32,32,32
# losses = fit_field(
#     hybrid, 
#     dataset, # ([1, num_points, 3], [1, num_points, 2])..  3 is coordinates.. 2 is for [occ, 1-occ]
#     occ_loss,
#     (64, 64, 64), # is used for sample_plot_field_fn to test hybrid model.
#     sample_plot_occ_field, 
#     total_steps=5_001, 
#     lr=1e-4
# )
# check_losses("Fitting voxel neural field", losses, 5e-2)
# exit(0)

from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections.abc import Mapping # added

def to_gpu(ob):
    if isinstance(ob, Mapping): # origin version: if isinstance(ob, collections.Mapping):
        return {k: to_gpu(v) for k, v in ob.items()}
    elif isinstance(ob, tuple):
        return tuple(to_gpu(k) for k in ob)
    elif isinstance(ob, list):
        return [to_gpu(k) for k in ob]
    else:
        try:
            return ob.cuda()
        except:
            return ob

def fit_inverse_graphics_representation(representation,
                                        renderer, 
                                        data_iter,
                                        img_resolution,
                                        total_steps=2001,
                                        lr=1e-4
                                        ):
    # Define how many steps we want to train for,
    # and how often we want to write the summaries:
    steps_til_summary = 100

    # We will use the "Adam" stochastic gradient descent optimizer,
    # with an empirically chosen learning rate of 1e-3.
    # torch.autograd.set_detect_anomaly(True)
    optim = torch.optim.Adam(lr=lr, params=representation.parameters())

    losses = []
    frames = []
    for step in range(total_steps):
        # Get the next batch of data and move it to the GPU
        input_dict, ground_truth = next(data_iter)
        input_dict = to_gpu(input_dict)
        ground_truth = to_gpu(ground_truth)

        # Compute the MLP output for the given input data and compute the loss
        with torch.enable_grad(): # added this
            rgb, depth = renderer(input_dict, representation)

            loss = ((rgb- ground_truth) ** 2).mean()

            # Accumulate the losses so that we can plot them later
            losses.append(loss.detach().cpu().numpy())
        
        

        # Every so often, we want to show what our model has learned.
        # It would be boring otherwise!
        if not step % steps_til_summary:
            print(f"Step {step}: loss = {float(loss.detach().cpu()):.5f}")
            fig, axes = plt.subplots(1, 3, figsize=(18, 6), squeeze=False)
            axes[0, 0].imshow(rgb.cpu().view(*img_resolution).detach().numpy())
            axes[0, 0].set_title("Trained MLP")
            axes[0, 1].imshow(ground_truth.cpu().view(*img_resolution).detach().numpy())
            axes[0, 1].set_title("Ground Truth")
            
            depth = depth.cpu().view(*img_resolution[:2]).detach().numpy()
            axes[0, 2].imshow(depth, cmap='Greys')
            axes[0, 2].set_title("Depth")
            
            for i in range(3):
                axes[0, i].set_axis_off()

            # plt.show()
            print('Saving image...')
            # plt.savefig('images/image{:05d}.png'.format(step))    
            plt.savefig('custom_images/image{:05d}.png'.format(step))
            
           
         
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        del input_dict, ground_truth, loss
        gc.collect()
        torch.cuda.empty_cache()

    # We can also plot the values of our loss function to see how the optimization
    # minimized the loss during training. This is given mostly for demonstration purposes.
    # In practice, we would want to monitor the loss during training.
    fig, axes = plt.subplots(1, 1, figsize=(8, 8), squeeze=False)
    axes[0, 0].plot(np.array(losses))
    plt.savefig('custom_loss')
    # plt.show()




cam2world = np.load('/content/cam2world.npy')
cam2world = np.load('cam2world.npy')
images = np.load('images.npy')
print(images.shape)
print(images[0, :3, :3, :])
print(images.max(), images.min())
print(cam2world.shape)

# cuda() 대신에 원본은 cuda()
cam2world = torch.Tensor(cam2world).cuda()
images = torch.tensor(images).cuda()
intrinsics = torch.tensor([[0.7, 0., 0.5],
                            [0., 0.7, 0.5],
                            [0., 0., 1.]]).cuda()

# # custom data
# from get_data import *
# images = get_images_data()
# # print(images.shape)
# # print(images[0, :3, :3, :])
# # print(images.max(), images.min())
# cam2world, intrinsics = get_cam2world_data()
# # print(cam2world.shape) # (53, 4, 4)

# cam2world = torch.Tensor(cam2world).cuda()
# images = torch.tensor(images).cuda()
# intrinsics = torch.tensor(intrinsics).float().cuda()


bunny_dataset = diff_rendering_dataset(images, cam2world)
# bunny_dataset = diff_rendering_custom_dataset(images, cam2world, intrinsics)
rf = RadianceField().cuda() # was .cuda()
renderer = VolumeRenderer(near=1.5, far=4.5, n_samples=128, white_back=True, rand=False).cuda() # was .cuda()

print("Running fit inverse graphics...")
fit_inverse_graphics_representation(rf, renderer, bunny_dataset, (128, 128, 3), lr=1e-3, total_steps=2_001)
# fit_inverse_graphics_representation(rf, renderer, bunny_dataset, (72, 128, 3), lr=1e-3, total_steps=2_001)

print("Saving trained model...")
PATH_renderer = 'renderer3.pth'
PATH_rf = 'rf3.pth'
# PATH_renderer = 'renderer_custom.pth'
# PATH_rf = 'rf_custom.pth'
torch.save(renderer.state_dict(), PATH_renderer)
torch.save(rf.state_dict(), PATH_rf)
