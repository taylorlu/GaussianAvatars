#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from typing import Union
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene import GaussianModel, FlameGaussianModel
from utils.sh_utils import eval_sh
from io import BytesIO
import numpy as np


def sh_basis(normal):
    """
    Calculate the spherical harmonics basis functions up to 4th order for a given normal vector.
    Args:
        normal (numpy.ndarray): The normal vector (should be normalized).
    
    Returns:
        numpy.ndarray: The SH basis values.
    """
    x, y, z = normal
    return np.array([
        0.28209479177387814,  # l=0, m=0
        
        -0.4886025119029199 * y,  # l=1, m=-1
        0.4886025119029199 * z,   # l=1, m=0
        -0.4886025119029199 * x,  # l=1, m=1

        1.0925484305920792 * x * y,  # l=2, m=-2
        -1.0925484305920792 * y * z, # l=2, m=-1
        0.31539156525252005 * (2 * z ** 2 - x ** 2 - y ** 2),  # l=2, m=0
        -1.0925484305920792 * x * z, # l=2, m=1
        0.5462742152960396 * (x ** 2 - y ** 2),  # l=2, m=2
        
        -0.5900435899266435 * (3 * x ** 2 - y ** 2) * y,  # l=3, m=-3
        2.890611442640554 * x * y * z,  # l=3, m=-2
        -0.4570457994644658 * (4 * z ** 2 - x ** 2 - y ** 2) * y,  # l=3, m=-1
        0.3731763325901154 * (2 * z ** 2 - 3 * x ** 2 - 3 * y ** 2) * z,  # l=3, m=0
        -0.4570457994644658 * (4 * z ** 2 - x ** 2 - y ** 2) * x,  # l=3, m=1
        1.445305721320277 * (x ** 2 - y ** 2) * z,  # l=3, m=2
        -0.5900435899266435 * (x ** 2 - 3 * y ** 2) * x  # l=3, m=3
    ])

def calculate_color(sh_coeffs, normal):
    """
    Calculate color based on SH coefficients and surface normal.
    
    Args:
        sh_coeffs (numpy.ndarray): Array of SH coefficients with shape (16, 3) (RGB).
        normal (numpy.ndarray): Normal vector of the surface point.

    Returns:
        numpy.ndarray: Color (R, G, B) calculated from SH lighting.
    """
    # Calculate the SH basis functions for the given normal
    basis = sh_basis(normal)
    
    # Sum the contribution of each SH coefficient
    color = np.zeros(3) + 0.5
    for i in range(len(basis)):
        color += basis[i] * sh_coeffs[i]
        
    return np.clip(color, 0, 1)


def render(viewpoint_camera, pc : Union[GaussianModel, FlameGaussianModel], pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, index=0):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    means3D_np = means3D.detach().cpu().numpy()
    scales_np = scales.detach().cpu().numpy()
    rotations_np = rotations.detach().cpu().numpy()
    shs_np = shs.detach().cpu().numpy()
    opacity_np = opacity.detach().cpu().numpy()

    buffer = BytesIO()

    for idx in range(means3D_np.shape[0]):
        position = np.array([means3D_np[idx][0], means3D_np[idx][1], means3D_np[idx][2]], dtype=np.float32)
        scales_ = np.array(
                [scales_np[idx][0], scales_np[idx][1], scales_np[idx][2]],
                dtype=np.float32,
            )

        rot_ = np.array(
            [rotations_np[idx][0], rotations_np[idx][1], rotations_np[idx][2], rotations_np[idx][3]],
            dtype=np.float32,
        )
        dir = position - np.array([0.0, 0.0, 1])
        dir /= np.linalg.norm(dir)
        color = np.concatenate([calculate_color(shs_np[idx], dir), opacity_np[idx]])

        # SH_C0 = 0.28209479177387814
        # color = np.array(
        #     [
        #         0.5 + SH_C0 * shs_np[idx][0][0],
        #         0.5 + SH_C0 * shs_np[idx][0][1],
        #         0.5 + SH_C0 * shs_np[idx][0][2],
        #         opacity_np[idx][0],
        #     ]
        # )
        buffer.write(position.tobytes())
        buffer.write(scales_.tobytes())
        buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
        buffer.write(
            ((rot_ / np.linalg.norm(rot_)) * 128 + 128)
            .clip(0, 255)
            .astype(np.uint8)
            .tobytes()
        )

    with open(f'D:/splat/splats/{index}.splat', "wb") as f:
        f.write(buffer.getvalue())

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
