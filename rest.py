import torch.nn.functional as F
import torch
from torch import nn
from typing import Callable, List, Optional, Tuple, Generator, Dict
import matplotlib.pyplot as plt
import numpy as np
import imageio
# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
#     torch.cuda.set_device(device)
# else:
#     device = torch.device("cpu")

device = torch.device("cpu")


def get_norm_pixel_coordinates(
    y_resolution: int,
    x_resolution: int,
    device: torch.device = torch.device('cpu')
    ):
    """
    Returns:
        xy_pix: 
    """
    x = torch.linspace(0, 1, steps=x_resolution, device=device)
    # x = torch.linspace(0, 1, steps=x_resolution, device=device)
    y = torch.linspace(0, 1, steps=y_resolution, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='xy') #xx, yy shape = y_resolution*x_resolution because I use 'xy'. x is col index, y is row index
    xy_pix = torch.stack((xx,yy), dim=-1).reshape(-1,2)

    return xy_pix


    # xs = torch.linspace(0, 1, x_resolution, device=device)
    # ys = torch.linspace(0, 1, y_resolution, device=device)
    # x, y = torch.meshgrid(xs, ys, indexing='xy')
    # xy_pix = torch.stack((x, y), dim=-1)
    # return xy_pix

def get_training_data(image: torch.Tensor): # image.shape = (B, H, W)
    """ 

    Returns:
        x: normalized pixel coordinate, generated using shape of image. (B=1, N, 2)
        b: image pixels (B=1, N, 1)

    """
  
    batch_size = 1
    y_resolution, x_resolution = image.shape

    x = get_norm_pixel_coordinates(y_resolution, x_resolution).reshape(batch_size, -1, 2).cuda() # xy_norm_grid.shape = (batchsize, y_reso * x_reso, 2)
    b = image.reshape(batch_size, -1, 1).cuda() # image pixel values
  
    return x, b





def sample_plot_img_field(
    field: nn.Module,
    image_size: Tuple[int, int],
    ax = None
    ):

    coords = get_norm_pixel_coordinates(*image_size).reshape(1, -1, 2).cuda()

  
    # use this when infer
    with torch.no_grad():
    # with torch.enable_grad():
        # Sample "field" with these coordinates
        model_out = field(coords).cuda()
    #########

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # ax.imshow(model_out.cpu().view(*image_size).detach().numpy())
    
    ax.imshow(model_out.cpu().view(*image_size).detach().numpy(), cmap='gray') # maybe try this? since 
    # gt image is gray scale?
    return model_out


def mse_loss(mlp_out, gt):
    return ((mlp_out - gt)**2).mean()



def fit_field(
    representation: nn.Module,
    data_iterator: Generator[Tuple[torch.Tensor, torch.Tensor], None, None], # maybe referring to x, b from previous cells?
    loss_fn,
    resolution: Tuple,
    sample_plot_field_fn,
    steps_til_summary = 500,
    total_steps=2001,
    lr=1e-3
    ):
    # We will use the "Adam" stochastic gradient descent optimizer,
    # with a learning rate of 1e-3.
    optim = torch.optim.Adam(lr=lr, params=representation.parameters())

    losses = []
    frames = []
    for step in range(total_steps):
    

        # Get the next batch of data and move it to the GPU
        # ([1, num_points, 3], [1, num_points, 2]) for voxel
        model_input, ground_truth = next(data_iterator) # previous cell x, b where x = B,N,2: normalized pixel coordinates
        # ground_truth = b where b is real pixel values of image
        
        # model_input = model_input.cuda()
        model_input = model_input.cuda()
        
        ground_truth = ground_truth.cuda()
        
        with torch.enable_grad():
          model_output = representation(model_input).cuda()
          loss = loss_fn(model_output, ground_truth).cuda()

       

        optim.zero_grad()
        loss.backward() # compute gradient
        optim.step()

        # Accumulate the losses so that we can plot them later
        losses.append(loss.detach().cpu().numpy())            

        # Every so often, we want to show what our model has learned.
        # It would be boring otherwise!
        if not step % steps_til_summary:
            print(f"Step {step}: loss = {float(loss.detach().cpu()):.4f}")
            frame = sample_plot_field_fn(representation, resolution) # resolution: 64,64,64
            frames.append(frame)
            # plt.show()
    f = 'video2.mp4'
    imageio.mimwrite(f, frames, fps=1, quality=7)


    return losses

def get_norm_voxel_coordinates(
    resolution_per_axis: Tuple[int, int, int],
    device: torch.device = torch.device('cpu')
    ):
    xyz_res = resolution_per_axis
    i, j, k = torch.meshgrid(torch.linspace(-1, 1, steps=xyz_res[0], device=device), 
                             torch.linspace(-1, 1, steps=xyz_res[1], device=device),
                             torch.linspace(-1, 1, steps=xyz_res[2], device=device))
    # i, j, k = torch.meshgrid(torch.linspace(-0.5, 0.5, steps=xyz_res[0], device=device), 
    #                          torch.linspace(-0.5, 0.5, steps=xyz_res[1], device=device),
    #                          torch.linspace(-0.5, 0.5, steps=xyz_res[2], device=device))

    xyz_vox = torch.stack([i.float(), j.float(), k.float()], dim=-1)
    return xyz_vox # xyz_vox.shape is maybe [i_res, j_res, k_res, 3]


def sphere_occupancy(
    x: torch.Tensor # shape is maybe size=(1, num_points, 3)
):
    '''Occupancy function of a 3D sphere with radius 0.5.
    Returns (1, 0) if x is occupied and (0, 1) otherwise.
    '''
    occ = (x.norm(dim=-1, keepdim=True) < 0.5).float() # 0.5 here is radius, not likelihood 
    return torch.cat((occ, 1-occ), dim=-1) # size=(1, num_points, 2)



def sample_plot_occ_field(
    field,
    voxel_resolution: Tuple[int, int, int],
    ax = None
    ):
    # frames = []
    #########
    
    # Get normalized pixel coordinates for image_size and reshape it to (1, N, 3)
    # where N is V_x, V_y, V_z are the dimensions defined in voxel_resolution.
    # coords = get_norm_voxel_coordinates(voxel_resolution).reshape(1, -1, 3).cuda()
    coords = get_norm_voxel_coordinates(voxel_resolution)[None].cuda() # (1, Vx, Vy, Vz, )
    # Infer the field on the coords
    with torch.no_grad():
        out = field(coords) # out size=(1, num_points, 2)

    # Reshape the output as (V_x, V_y, V_z, 2) for V_x, V_y, V_z are the 
    # dimensions of the voxel grid.
    out = out.reshape(*voxel_resolution, 2).cuda() 
    #########

    fig, axes = plt.subplots(1, 3, figsize=(18,6))
    half_res = voxel_resolution[0] // 2
    # half = coords.shape[1] // 2
    # coords = coords.squeeze()
    # print(f'coords shape: {coords.shape}')
    axes[0].imshow(out[:, :, half_res, 0].squeeze().cpu()) # 0 here is occupancy value. [1] is 1-occupancy
    axes[1].imshow(out[half_res, :, :, 0].squeeze().cpu())
    axes[2].imshow(out[:, half_res, :, 0].squeeze().cpu()) 
    # axes[0].imshow(coords[:, :, half, 0].squeeze().cpu())
    # axes[1].imshow(coords[half, :, :, 0].squeeze().cpu())
    # axes[2].imshow(coords[:, half, :, 0].squeeze().cpu()) 
    
    import cv2
    print('Concatenating images...')
   
    print(out.shape)
    from skimage import img_as_ubyte
    print(f'min and max: {out.max()}, {out.min()}')
    concat_imgs = np.concatenate([img_as_ubyte(out[:, :, half_res, 0].cpu()), 
                                img_as_ubyte(out[half_res, :, :, 0].cpu()), 
                                img_as_ubyte(out[:, half_res, :, 0].cpu())], axis=-1)


    # frames.append(concat_imgs)
    return concat_imgs
    # import imageio
    

    # plt.show()


def sphere_generator(voxel_resolution: Tuple[int, int, int]):     
    while True:        
        # coords = torch.rand(size=(1, num_points, 3)) # 0~1   
        # coords = (coords - 0.5) * 2 # -0.5 ~ 0.5  -> -1 ~ 1

        coords_for_gt = get_norm_voxel_coordinates(voxel_resolution).reshape(1, -1, 3).cuda()

        yield coords_for_gt, sphere_occupancy(coords_for_gt) # coords.shape = [1, num_points, 3], sphere_occupancy(coords).shape = [1, num_points, 2]
                                                             # coords_for_gt is of 3d coordinates and sphere_occupancy is occupancy values of each position

def sphere_generator_random(voxel_resolution: Tuple[int, int, int]):     
    num_points = voxel_resolution[0]*voxel_resolution[1]*voxel_resolution[2]
    while True:
        # num_points = voxel_resolution[0] * voxel_resolution[1] * voxel_resolution[2]
        coords = torch.rand(size=(1, num_points, 3)).reshape(1,-1,3) # 0~1   
        coords = (coords - 0.5) * 2 # 0 ~ 1 -> -0.5 ~ 0.5  -> -1 ~ 1 .  coords are coordinates in [-1,1]**3 cubic
        
        # coords = get_norm_voxel_coordinates(voxel_resolution).reshape(1, -1, 3).cuda()        
        # coords = get_norm_voxel_coordinates(voxel_resolution)[None].cuda()        
       
        coords = coords.reshape(1, *voxel_resolution, 3)
        yield coords, sphere_occupancy(coords) # coords.shape = [1, num_points, 3], sphere_occupancy(coords).shape = [1, num_points, 2]



def plot_pointcloud(
    vertices, 
    alpha=.5, 
    title=None, 
    max_points=10_000, 
    xlim=(-1, 1), 
    ylim=(-1, 1),
    zlim=(-1, 1)
    ):
    """Plot a pointcloud tensor of shape (N, coordinates)
    """
    vertices = vertices.cpu()

    assert len(vertices.shape) == 2
    N, dim = vertices.shape
    assert dim==2 or dim==3

    if N > max_points:
        vertices = np.random.default_rng().choice(vertices, max_points, replace=False)
    fig = plt.figure(figsize=(6,6))
    if dim == 2:
        ax = fig.add_subplot(111)
    elif dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlabel("z")
        ax.set_zlim(zlim)
        ax.view_init(elev=120., azim=270) # azim starts at x axis. azim > 0 = rotate to y pos dir. top of camera is heading z pos axis

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.scatter(*vertices.T, alpha=alpha, marker=',', lw=.5, s=1, color='black')
    plt.show(fig)

def occ_loss(mlp_out, gt):
  return F.binary_cross_entropy_with_logits(mlp_out, gt)