from einops import rearrange, repeat
import numpy as np
import imageio
import collections
import torch

def homogenize_points(points: torch.Tensor):
    """Appends a "1" to the coordinates of a (batch of) points of dimension DIM.

    Args:
        points: points of shape (..., DIM)

    Returns:
        points_hom: points with appended "1" dimension.
    """
    ones = torch.ones_like(points[..., :1], device=points.device)
    return torch.cat((points, ones), dim=-1)


def homogenize_vecs(vectors: torch.Tensor):
    """Appends a "0" to the coordinates of a (batch of) vectors of dimension DIM.

    Args:
        vectors: vectors of shape (..., DIM)

    Returns:
        vectors_hom: points with appended "0" dimension.
    """
    zeros = torch.zeros_like(vectors[..., :1], device=vectors.device)
    return torch.cat((vectors, zeros), dim=-1)


def unproject(
    xy_pix: torch.Tensor, 
    z: torch.Tensor, 
    intrinsics: torch.Tensor
    ) -> torch.Tensor:
    """Unproject (lift) 2D pixel coordinates x_pix and per-pixel z coordinate
    to 3D points in camera coordinates.

    Args:
        xy_pix: 2D pixel coordinates of shape (..., 2)
        z: per-pixel depth, defined as z coordinate of shape (..., 1) 
        intrinscis: camera intrinscics of shape (..., 3, 3)

    Returns:
        xyz_cam: points in 3D camera coordinates. shape is (..., 3)
    """
    xy_pix_hom = homogenize_points(xy_pix) # xy_pix_hom.shape = (..., 3)
    xyz_cam = torch.einsum('...ij,...j->...i', intrinsics.inverse(), xy_pix_hom)
    xyz_cam *= z
    return xyz_cam
    

def transform_world2cam(xyz_world_hom: torch.Tensor, cam2world: torch.Tensor) -> torch.Tensor:
    """Transforms points from 3D world coordinates to 3D camera coordinates.

    Args:
        xyz_world_hom: homogenized 3D points of shape (..., 4)
        cam2world: camera pose of shape (..., 4, 4)

    Returns:
        xyz_cam: points in camera coordinates.
    """
    world2cam = torch.inverse(cam2world)
    return transform_rigid(xyz_world_hom, world2cam)


def transform_cam2world(xyz_cam_hom: torch.Tensor, cam2world: torch.Tensor) -> torch.Tensor:
    """Transforms points from 3D world coordinates to 3D camera coordinates.

    Args:
        xyz_cam_hom: homogenized 3D points of shape (..., 4)
        cam2world: camera pose of shape (..., 4, 4)

    Returns:
        xyz_world: points in camera coordinates.
    """
    return transform_rigid(xyz_cam_hom, cam2world)


def transform_rigid(xyz_hom: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Apply a rigid-body transform to a (batch of) points / vectors.

    Args:
        xyz_hom: homogenized 3D points of shape (..., 4)
        T: rigid-body transform matrix of shape (..., 4, 4)

    Returns:
        xyz_trans: transformed points. shape is (..., 4)
    """ 
    return torch.einsum('...ij, ...j -> ...i', T, xyz_hom)


def get_camera_ray_directions(xy_pix:torch.Tensor,
                              intrinsics:torch.Tensor) -> torch.Tensor:
    """Returns the 3D ray directions for xy_pix coordinates in camera coordinates.
    """
    p = unproject(xy_pix, 
                  torch.ones_like(xy_pix[..., :1], device=xy_pix.device), 
                  intrinsics=intrinsics)
    return torch.nn.functional.normalize(p, dim=-1) # along column


def get_unnormalized_cam_ray_directions(xy_pix:torch.Tensor,
                                        intrinsics:torch.Tensor) -> torch.Tensor:
    '''Returns output of unproject without normalizing each direction. Shape is
    shape is (..., 3).
    '''
    return unproject(xy_pix, torch.ones_like(xy_pix[..., :1], device=xy_pix.device),  intrinsics=intrinsics)
