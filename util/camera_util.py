from typing import Sequence
import jax.numpy as jnp
import numpy as np
import jax
import einops
import util.transform_util as tutil

def default_intrinsic(pixel_size:Sequence[int]):
    return jnp.array([pixel_size[1], pixel_size[0], pixel_size[1], pixel_size[0], 0.5*pixel_size[1], 0.5*pixel_size[0]])

def pixel_ray(pixel_size:Sequence[int], cam_pos:jnp.ndarray, cam_quat:jnp.ndarray, 
                intrinsic:jnp.ndarray, near:float, far:float, coordinate:str='opengl'):
    '''
    bachable
    pixel_size : (2,) i j order
    cam_pos : (... 3) camera position
    cam_quat : (... 4) camera quaternion
    intrinsic : (... 6) camera intrinsic
    
    coordinate : 'opengl' or 'open3d' - oepngl: forward direction is z minus axis / open3d: forward direction is z plus
    '''
    cam_zeta = intrinsic[...,2:]
    zeros = jnp.zeros_like(cam_zeta[...,0])
    ones = jnp.ones_like(cam_zeta[...,0])
    K_mat = jnp.stack([jnp.stack([cam_zeta[...,0], zeros, cam_zeta[...,2]],-1),
                        jnp.stack([zeros, cam_zeta[...,1], cam_zeta[...,3]],-1),
                        jnp.stack([zeros,zeros,ones],-1)],-2)

    # pixel= PVM (colomn-wise)
    # M : points
    # V : inv(cam_SE3)
    # P : Z projection and intrinsic matrix  
    x_grid_idx, y_grid_idx = jnp.meshgrid(jnp.arange(pixel_size[1])[::-1], jnp.arange(pixel_size[0])[::-1])
    pixel_pnts = jnp.concatenate([x_grid_idx[...,None], y_grid_idx[...,None], jnp.ones_like(y_grid_idx[...,None])], axis=-1)
    pixel_pnts = jnp.array(pixel_pnts, dtype=jnp.float32)
    K_mat_inv = jnp.linalg.inv(K_mat)
    pixel_pnts = jnp.matmul(K_mat_inv[...,None,None,:,:], pixel_pnts[...,None])[...,0]
    if coordinate == 'opengl':
        pixel_pnts = pixel_pnts.at[...,-1].set(-pixel_pnts[...,-1])
        pixel_pnts = pixel_pnts[...,::-1,:]
    rays_s_canonical = pixel_pnts * near
    rays_e_canonical = pixel_pnts * far

    # cam SE3 transformation
    rays_s = tutil.pq_action(cam_pos[...,None,None,:], cam_quat[...,None,None,:], rays_s_canonical)
    rays_e = tutil.pq_action(cam_pos[...,None,None,:], cam_quat[...,None,None,:], rays_e_canonical)
    ray_dir = rays_e - rays_s
    ray_dir_normalized = tutil.normalize(ray_dir)

    return rays_s, rays_e, ray_dir_normalized

def pbfov_to_intrinsic(img_size, fov_deg):
    # assert img_size[0] == img_size[1]
    fov_rad = fov_deg * np.pi/180.0
    Fy = img_size[0]*0.5/(np.tan(fov_rad*0.5))
    # Fx = img_size[1]*0.5/(np.tan(fov_rad*0.5))
    Fx = Fy
    Cx = img_size[1]*0.5
    Cy = img_size[0]*0.5
    return (img_size[1], img_size[0], Fx, Fy, Cx, Cy)

def intrinsic_to_fov(intrinsic):
    img_size_xy = intrinsic[...,:2]
    fovs = np.arctan(intrinsic[...,1]/intrinsic[...,3]*0.5)*2
    return fovs, img_size_xy[...,0] / img_size_xy[...,1]

def intrinsic_to_pb_lrbt(intrinsic, near):
    if isinstance(intrinsic, list) or isinstance(intrinsic, tuple):
        intrinsic = np.array(intrinsic)
    pixel_size = intrinsic[...,:2]
    fx = intrinsic[...,2]
    fy = intrinsic[...,3]
    cx = intrinsic[...,4]
    cy = intrinsic[...,5]
    
    halfx_px = pixel_size[...,0]*0.5
    center_px = cx - halfx_px
    right_px = center_px + halfx_px
    left_px = center_px - halfx_px

    halfy_px = pixel_size[...,1]*0.5
    center_py = cy - halfy_px
    bottom_px = center_py - halfy_px
    top_px = center_py + halfy_px

    # lrbt = np.stack([left_px, right_px, bottom_px, top_px], axis=-1)
    # lrbt = lrbt/np.stack([fx,fx,fy,fy], axis=-1) * near
    return left_px/fx*near, right_px/fx*near, bottom_px/fy*near, top_px/fy*near


def intrinsic_to_Kmat(intrinsic):
    zeros = jnp.zeros_like(intrinsic[...,2])
    return jnp.stack([jnp.stack([intrinsic[...,2], zeros, intrinsic[...,4]], -1),
                jnp.stack([zeros, intrinsic[...,3], intrinsic[...,5]], -1),
                jnp.stack([zeros, zeros, jnp.ones_like(intrinsic[...,2])], -1)], -2)
    

def global_pnts_to_pixel(intrinsic, cam_posquat, pnts):
    '''
    intrinsic, cam_posquat : (... NR ...)
    pnts : (... NS 3)
    '''
    pixel_size = intrinsic[...,:2]
    pnt_img_pj = tutil.pq_action(*tutil.pq_inv(*cam_posquat), pnts) # (... NS NR 3)
    kmat = intrinsic_to_Kmat(intrinsic)
    px_coord_xy = jnp.einsum('...ij,...j', kmat[...,:2,:2], pnt_img_pj[...,:2]/(-pnt_img_pj[...,-1:])) + kmat[...,:2,2]
    out_pnts = jnp.logical_or(jnp.any(px_coord_xy<0, -1), jnp.any(px_coord_xy>=pixel_size, -1))
    px_coord_xy = jnp.clip(px_coord_xy, 0.001, pixel_size-0.001)
    px_coord_ij = jnp.stack([pixel_size[...,1]-px_coord_xy[...,1], px_coord_xy[...,0]], -1)
    return px_coord_ij, out_pnts
    # px_coord = px_coord.astype(jnp.float32)
    # px_coord = jnp.stack([-px_coord[...,1], px_coord[...,0]] , -1)# xy to ij

def cam_info_to_render_params(cam_info):
    cam_posquat, intrinsic = cam_info

    return dict(
        intrinsic=intrinsic,
        pixel_size=jnp.c_[intrinsic[...,1:2], intrinsic[...,0:1]].astype(jnp.int32),
        camera_pos=cam_posquat[...,:3],
        camera_quat=cam_posquat[...,3:],
    )


def pcd_from_depth_np(depth, intrinsic, coordinate='opengl', visualize=False):
    if depth.shape[-1] != 1:
        depth = depth[...,None]
    
    pixel_size = (int(intrinsic[0]), int(intrinsic[1]))
    
    xgrid, ygrid = np.meshgrid(np.arange(pixel_size[1]), np.arange(pixel_size[0]), indexing='xy')
    xygrid = np.stack([xgrid, ygrid], axis=-1)
    xy = (xygrid - intrinsic[4:6]) * depth / intrinsic[2:4]

    xyz = np.concatenate([xy, depth], axis=-1)

    if coordinate=='opengl':
       xyz =  np.stack([xyz[...,0], -xyz[...,1], -xyz[...,2]], axis=-1)

    if visualize:
        import open3d as o3d
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(xyz.reshape(-1,3))
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd_o3d, mesh_frame])

    return xyz



def pcd_from_depth(depth, intrinsic, pixel_size, coordinate='opengl', visualize=False):
    if depth.shape[-1] != 1:
        depth = depth[...,None]

    xgrid, ygrid = np.meshgrid(np.arange(pixel_size[1]), np.arange(pixel_size[0]), indexing='xy')
    xygrid = np.stack([xgrid, ygrid], axis=-1)
    xy = (xygrid - intrinsic[...,None,None,4:6]) * depth / intrinsic[...,None,None,2:4]

    xyz = jnp.concatenate([xy, depth], axis=-1)

    if coordinate=='opengl':
       xyz =  jnp.stack([xyz[...,0], -xyz[...,1], -xyz[...,2]], axis=-1)

    if visualize:
        import open3d as o3d
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(xyz.reshape(-1,3))
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd_o3d, mesh_frame])

    return xyz