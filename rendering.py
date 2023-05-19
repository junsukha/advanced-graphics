from geometry import *
import torch
from utils import *
import torch.nn.functional as F
import torch
from torch import nn
from typing import Callable, List, Optional, Tuple, Generator, Dict
import matplotlib.pyplot as plt
import numpy as np
from rest import *
from hybrid import *

# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
#     torch.cuda.set_device(device)
# else:
#     device = torch.device("cpu")

device = torch.device("cpu")


class RadianceField(nn.Module):
    def __init__(self):
        super().__init__()
       
        # In a member "self.scene_rep", store a HybridVoxelNeuralField
        # with a sidelength of 64, 64 hidden features, and 64 output features.
        self.scene_rep = HybridVoxelNeuralField(resolution_per_dim=(64,64,64), feature_dim=64, out_dim=64)
            
        # Write a (ReLU, linear, ReLU) MLP "self.sigma" that will take as input
        # the output of the ground plan field and output a scalar density.
        self.sigma = nn.Sequential(nn.ReLU(),
                                   nn.Linear(in_features=64, out_features=1),
                                   nn.ReLU())

        # Write a (ReLU, linear, Sigmoid) MLP "self.radiance" that will take as input
        # the output of the ground plan field and output a 3 channel RGB.
        self.radiance = nn.Sequential(nn.ReLU(inplace=True),
                                   nn.Linear(in_features=64, out_features=3),
                                   nn.Sigmoid())

        # Apply init_weights_normal to both mlp models
        self.sigma.apply(init_weights_normal)
        self.radiance.apply(init_weights_normal)
   

    def forward(
        self, 
        xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
      '''
      Queries the representation for the density and color values.
      xyz is a tensor of shape (batch_size, num_samples, 3)
      '''
  
      # Do a forward pass through the scene representation.
      features = self.scene_rep(xyz)
      # Do a forward pass through both the self.sigma and self.radiance MLPs
      # to yield sigma and color.
      sigma = self.sigma(features)
      # print(f'radience field. sigma is cuda: {sigma.is_cuda}')
      rad = self.radiance(features)
      return sigma, rad
    
def get_opencv_pixel_coordinates(
    y_resolution: int,
    x_resolution: int,
    device: torch.device = torch.device('cpu')
    ):
    """For an image with y_resolution and x_resolution, return a tensor of pixel coordinates
    normalized to lie in [0, 1], with the origin (0, 0) in the top left corner,
    the x-axis pointing right, the y-axis pointing down, and the bottom right corner
    being at (1, 1).

    Returns:
        xy_pix: a meshgrid of values from [0, 1] of shape 
                (y_resolution, x_resolution, 2)
    """
    # original version
    i, j = torch.meshgrid(torch.linspace(0, 1, steps=x_resolution, device=device), 
                          torch.linspace(0, 1, steps=y_resolution, device=device))

    xy_pix = torch.stack([i.float(), j.float()], dim=-1).permute(1, 0, 2) # before permute, i,j,2 . after permute: j, i, 2 (width, height, 2)

    # Another version
    # x = torch.linspace(0, 1, x_resolution)
    # y = torch.linspace(0, 1, y_resolution)

    # xx , yy = torch.meshgrid((x, y), indexing='xy')
    # xy_pix = torch.stack((xx, yy), dim=-1)

    return xy_pix
    
def diff_rendering_custom_dataset(images, cam2world, intrinsics):
    '''Generates an iterator from a tensor of images and a tensor of cam2world matrices.
    Yield *one random image per iteration*
    cam2world.shape = (b, 4, 4)
    images.shape = (b, w, h, 4)
    '''
    image_resolution = images.shape[1:3] # image.shape = [100, 128, 128, 4]
    intrinsics = intrinsics.to(images.device)

    x_pix = get_opencv_pixel_coordinates(*image_resolution) # (width, height, 2)
    x_pix = x_pix.reshape(1, -1, 2).to(images.device) # [1, num_points, 2] where num_points = y_resolution * x_resolution
    # x_pix = x_pix.reshape(1, images.shape[1], images.shape[2], 2)
    while True:
        idx = np.random.randint(low=0, high=len(cam2world))
        # idx = 80 # delete this
        c2w = cam2world[idx:idx+1] # c2w shape = [1,4,4]
        ground_truth = images[idx:idx+1] # get an image of which camera's extrinsic is c2w.. [1, 128, 128, 4]
        intrinsic = intrinsics[idx:idx+1]
        model_input = {'cam2world': c2w, 
                        'intrinsics': intrinsic, 
                        'x_pix': x_pix}
        yield model_input, ground_truth[..., :3].view(-1, 3) # [num_points, 3]

def diff_rendering_dataset(images, cam2world):
    '''Generates an iterator from a tensor of images and a tensor of cam2world matrices.
    Yield *one random image per iteration*
    cam2world.shape = (b, 4, 4)
    images.shape = (b, w, h, 4)
    '''
    image_resolution = images.shape[1:3] # image.shape = [100, 128, 128, 4]
    intrinsics = torch.tensor([[0.7, 0., 0.5], # Copied from last assignment
                               [0., 0.7, 0.5],
                               [0., 0., 1.]]).to(images.device)

    x_pix = get_opencv_pixel_coordinates(*image_resolution) # (width, height, 2)
    x_pix = x_pix.reshape(1, -1, 2).to(images.device) # [1, num_points, 2] where num_points = y_resolution * x_resolution
    # x_pix = x_pix.reshape(1, images.shape[1], images.shape[2], 2)
    while True:
        idx = np.random.randint(low=0, high=len(cam2world))
        # idx = 80 # delete this
        c2w = cam2world[idx:idx+1] # c2w shape = [1,4,4]
        ground_truth = images[idx:idx+1] # get an image of which camera's extrinsic is c2w.. [1, 128, 128, 4]
        model_input = {'cam2world': c2w, 
                        'intrinsics': intrinsics, 
                        'x_pix': x_pix}
        yield model_input, ground_truth[..., :3].view(-1, 3) # [num_points, 3]


def get_world_rays(xy_pix: torch.Tensor, 
                   intrinsics: torch.Tensor, #shape = 3,3
                   cam2world: torch.Tensor,
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
   
    # Get camera origin of camera 1
    # print(f'cam2world: {cam2world.shape}')
    cam_origin_world = cam2world[:, :3 ,-1:].permute(0,2,1) # size should be (batch, 3, 1)
    # print(f'cam_origin_world.shape: {cam_origin_world.shape}')

    # Get ray directions in cam coordinates
    ray_dirs_cam = get_unnormalized_cam_ray_directions(xy_pix, intrinsics) # shape (batch_size,  2)
    # print(ray_dirs_cam.shape)

    # Homogenize ray directions
    rd_cam_hom = homogenize_vecs(ray_dirs_cam) # shape (..., 3)
    # print(rd_cam_hom.shape)

    # Transform ray directions to world coordinates
    rd_world_hom = transform_cam2world(rd_cam_hom, cam2world) 
    # print(rd_world_hom.shape)

    # Tile the ray origins to have the same shape as the ray directions.
    # Currently, ray origins have shape (batch, 3), while ray directions have shape
    cam_origin_world = cam_origin_world[..., :3].clone()

    # print(cam_origin_world.shape)
    # print(cam_origin_world)
    # print(rd_world_hom[..., :3])
    # Return tuple of cam_origins, ray_world_directions
    # print(f'cam_origin_world.shape: {cam_origin_world.shape}')
    # print(f'rd_world_hom[..., :3].shape: {rd_world_hom[..., :3].shape}')
    # print(cam_origin_world)
    return cam_origin_world, rd_world_hom[..., :3].clone()



def sample_points_along_rays(
    near_depth: float,
    far_depth: float,
    num_samples: int,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
) -> torch.Tensor:
    '''Returns 3D coordinates of points along the camera rays defined by ray_origin
    and ray_directions. Depth values are uniformly sampled between the near_depth and
    the far_depth.

    Parameters:
        near_depth: float. The depth at which we start sampling points.
        far_depth: float. The depth at which we stop sampling points.
        num_samples: int. The number of depth samples between near_depth and far_depth.
        ray_origins: Tensor of shape (batch_size, num_rays, 3). The origins of camera rays. My note: I think
        this should be (batch_size, num_origins, 3) because we have num_rays for each num_origins.
        ray_directions: Tensor of shape (batch_size, num_rays, 3). The directions of camera rays.

    Returns:
        Tuple of (pts, z_vals).
        pts: tensor of shape (batch_size, num_rays, num_samples, 3). 3D points uniformly sampled
                between near_depth and far_depth
        z_vals: tensor of shape (num_samples) of depths linearly spaced between near and far plane.
    '''
    ######
    # print(f'ray_origins_shape: {ray_origins.shape}')
    # print(f'ray_directions.shape {ray_directions.shape}')
    # Compute a linspace of num_samples depth values beetween near_depth and far_depth.
    z_vals = torch.linspace(near_depth, far_depth, num_samples)

    # Using the ray_origins, ray_directions, generate 3D points along
    # the camera rays according to the z_vals.

    block = []
    for b in range(len(ray_directions)):
      for z in z_vals:
        block.append((ray_origins[b].clone() + z * ray_directions[b].clone()).tolist()) # (num_rays, 3)

    batch_size = ray_origins.shape[0]
    num_rays = ray_directions.shape[1]

    # print(f'block.shape: {torch.tensor(block).shape}')

    pts = torch.tensor(block).reshape(batch_size, num_rays, num_samples, 3) # num_rays is basically h*w 
    ######
    # print(pts)
    return pts, z_vals


def volume_integral(
    z_vals: torch.tensor,
    sigmas: torch.tensor,
    radiances: torch.tensor
) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    '''Computes the volume rendering integral.

    Parameters:
        z_vals: tensor of shape (num_samples) of depths linearly spaced between near and far plane.
        sigmas: tensor of shape (batch_size, num_rays, num_samples, 1). Densities 
            of points along rays.
        radiances: tensor of shape (batch_size, num_rays, num_samples, 3). Emitted
            radiance of points along rays.

    Returns:
        Tuple of (rgb, depth_map, weights).
        rgb: Tensor of shape (batch_size, num_rays, 3). Total radiance observed by rays.
            Computed of weighted sum of radiances along rays.
        depth_map: Tensor of shape (batch_size, num_rays, 1). Expected depth of each ray.
            Computed as weighted sum of z_vals along rays.
    '''
    
    # Compute the deltas in depth between the points.
    # i'm assuming all distances are the same. 
    dists = ((z_vals[-1].clone()-z_vals[0].clone())/(len(z_vals)-1)).repeat(len(z_vals))# len(dists) = len(z_vals) 
    dists[-1] = 0

    # Compute the alpha values from the densities and the dists.
    # Tip: use torch.einsum for a convenient way of multiplying the correct 
    # dimensions of the sigmas and the dists.
    # print(f'sigmas is cuda: {sigmas.is_cuda}')
    # print(f'dists is cuda: {dists.is_cuda}')
    sigmas_dists_mul = torch.einsum('bijk, j -> bijk', sigmas, dists.cuda())
    alpha = 1- torch.exp(-sigmas_dists_mul) # alpha.shape = (batch_size, num_rays, num_samples, 1)

    # Compute the Ts from the alpha values. Use torch.cumprod.
    Ts = torch.cumprod(1-alpha, dim=2) # Ts.shape = (batch_size, num_rays, num_samples, 1)
    # print(1-alpha)
    # print(Ts)
    # Compute the weights from the Ts and the alphas.
    weights = (alpha * Ts) # weights.shape = (batch_size, num_rays, num_samples, 1)
    
    # Compute the pixel color as the weighted sum of the radiance values.
    # (weights * radiances).shape = (batch_size, num_rays, num_samples, 3)
    # rgb.shape should be (batch_size, num_rays, 3) I think?
    
    rgb = torch.sum(weights * radiances, dim=2) # radiances = (batch_size, num_rays, num_samples, 3)..   

    # Compute the depths as the weighted sum of z_vals.
    # Tip: use torch.einsum for a convenient way of computing the weighted sum,
    # without the need to reshape the z_vals.
    # print(f'weights is cuda: {weights.is_cuda}')
    # print(f'z_vals is cuda: {z_vals.is_cuda}')
    depth_map = torch.sum(torch.einsum('bijk, j -> bijk', weights, z_vals.cuda()), dim=2)

    # print(f'weights: {weights}')
    # print(f'rgb: {rgb}')
    # print(depth_map)
    return rgb, depth_map, weights


class VolumeRenderer(nn.Module):
    def __init__(self, near, far, n_samples=32, white_back=True, rand=False):
        super().__init__()
        self.near = near
        self.far = far
        self.n_samples = n_samples
        self.white_back = white_back
        self.rand = rand

    def unpack_input_dict(self, 
                         input_dict:Dict[str, torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c2w = input_dict['cam2world']
        intrinsics = input_dict['intrinsics']
        x_pix = input_dict['x_pix'] # x_pix are query points
        return c2w, intrinsics, x_pix

    def forward(
        self, 
        input_dict: Dict[str, torch.Tensor],
        radiance_field: nn.Module
        ) -> Tuple[torch.tensor, torch.tensor]:
        """      
        Params:
            input_dict: Dictionary with keys 'cam2world', 'intrinsics', and 'x_pix'
            radiance_field: nn.Module instance of the radiance field we want to render.

        Returns:
            Tuple of rgb, depth_map
            rgb: for each pixel coordinate x_pix, the color of the respective ray.
            depth_map: for each pixel coordinate x_pix, the depth of the respective ray. 
        """
        cam2world, intrinsics, x_pix = self.unpack_input_dict(input_dict)
         # x_pix.shape = (batch_size, num_rays, 2). but.. 
        batch_size, num_rays = x_pix.shape[0], x_pix.shape[1] # x_pix.shape[1] is num_points.. i.e., w*h, i.e, # of rays
        # batch_size, nun_rays = x_pix.shape[0], x_pix.shape[1] * x_pix.shape[2]
        # print(f'x_pix shape: {x_pix.shape}')
        # Compute the ray directions in world coordinates.
        # Use the function get_world_rays.
        # ros (ray origins) = cam origin world coord.  shape = (batch_size, num_rays, 3)
        # we're not doing view dependent. only one camera origin. so cam2world.shape = [1,4,4]
        # rds (ray directions)= ray direction world coord hom. shape = (batch_size, num_rays, 3)
        ros, rds = get_world_rays(x_pix, intrinsics, cam2world)

        # print(f'ros shape: {ros.shape}')
        # print(f'rds shape: {rds.shape}')
        # Generate the points along rays and their depth values
        # Use the function sample_points_along_rays.
        # pts shape (batch_size, num_rays, num_samples, 3)
        # in this ex, ros shape = (1,3), rds shape = (1, 16384, 3)
        pts, z_vals = sample_points_along_rays(self.near, self.far, self.n_samples, ros, rds)

        # if self.rand:
        #     pts[..., -1:] += torch.rand_like(pts[..., -1:]) * (self.far-self.near)/self.n_samples

        # Reshape pts to (batch_size, -1, 3).
        pts = pts.reshape(batch_size, -1, 3)
        # pts = pts.reshape(batch_size, *x_pix.shape, 3)
        
        pts = pts.reshape(batch_size, 128, 128, self.n_samples, 3) # 128, 128 is image size generated by buuny_dataset. self.n_samples도 128임.
        # pts = pts.reshape(batch_size, 72, 128, self.n_samples, 3) # 128, 128 is image size generated by buuny_dataset. self.n_samples도 128임.


        # Sample the radiance field with the points along the rays.
        # radiance_field will infer sigma (density) and radiance for each sample point for each ray
        # radiance_field is basically a info box of scene; it knows/infers how given scene looks like
        sigma, rad = radiance_field(pts) # this uses features of voxelgrid

        # Reshape sigma and rad back to (batch_size, num_rays, self.n_samples, -1)
        sigma = sigma.reshape(batch_size, num_rays, self.n_samples, -1) # num_rays : h * w
        rad = rad.reshape(batch_size, num_rays, self.n_samples, -1) # self.n_samples : d

        # Compute pixel colors, depths, and weights via the volume integral.
        rgb, depth_map, weights = volume_integral(z_vals, sigma, rad)
        ##########

        if self.white_back:
            accum = weights.sum(dim=-2)
            rgb = rgb + (1. - accum)

        return rgb, depth_map