import jax.numpy as jnp
import flax.linen as nn
import jax
import ray
import numpy as np
import matplotlib.pyplot as plt
import einops
from functools import partial
import optax
import os, sys
import shutil
import datetime
import pickle
import pybullet as p
from PIL import Image
import glob
import pkgutil

import util.transform_util as tutil
import util.train_util as trutil
import util.camera_util as cutil
import scene_gen as sg

pixel_size = (100,100)
env_no = 10
data_limit = 100000
quat_nsample = 4096
seed = 0
batch_size = 512
inner_itr = 4
log_path = 'logs'
# mesh_name = 'can'
# mesh_name = 'spam'
# mesh_name = 'hair_dryer'
# mesh_name = 'obstacle2'
mesh_name = 'hammer2'
debug = False
datagen = True

jkey = jax.random.PRNGKey(seed)
quat_samples = tutil.qrand((quat_nsample,), jax.random.PRNGKey(0))
_, jkey = jax.random.split(jkey)


def ang_dif_from_quat(quat1, quat2):
    ang = jnp.linalg.norm(tutil.q2aa(tutil.qmulti(tutil.qinv(quat1), quat2)), axis=-1)
    ang = jnp.min(jnp.abs(jnp.stack([ang, np.pi*2 + ang, -np.pi*2 + ang], axis=-1)), axis=-1)
    return ang

def get_quat_idx(quat, projection=False):
    # option 1
    idx = jnp.argmin(jnp.sum((tutil.q2R(quat_samples) - tutil.q2R(quat[...,None,:]))**2, axis=(-1,-2)), axis=-1)
    # option 2
    # idx = jnp.argmin(ang_dif_from_quat(quat_samples, quat[...,None,:]), axis=-1)
    if projection:
        # quat = jnp.take_along_axis(quat_samples, idx[...,None,None], axis=-2)
        # quat = jnp.squeeze(quat, axis=-2)
        quat = jnp.take_along_axis(quat_samples, idx[...,None], axis=-2)
        return quat
    else:
        return idx

## quaternion test ##
quat = tutil.qrand((100000,))
quat2 = get_quat_idx(quat, True)
fn = jnp.sum((tutil.q2R(quat) - tutil.q2R(quat2))**2, axis=(-1,-2))
angdif = jnp.linalg.norm(tutil.q2aa(tutil.qmulti(tutil.qinv(quat), quat2)), axis=-1)
angdif2 = ang_dif_from_quat(quat, quat2)
## quaternion test ##

rb = trutil.replay_buffer(data_limit=data_limit, dataset_dir=glob.glob(os.path.join('dataset', mesh_name, '*.pkl')))
if datagen:
    scene_gen_ray = [ray.remote(sg.SceneGen).remote(pixel_size=pixel_size, meshname=mesh_name) for _ in range(env_no)]
    dataset = ray.get([sg.gen_dataset_itr.remote(itr=400) for sg in scene_gen_ray])
    dataset = jax.tree_map(lambda *x: np.concatenate(x, axis=0), *dataset)
    rb.push(dataset)
else:
    rb.load()
rb.set_val_data()
if datagen:
    scene_gen_ray = [ray.remote(sg.SceneGen).remote(pixel_size=pixel_size, meshname=mesh_name) for _ in range(env_no)]
    dataset = ray.get([sg.gen_dataset_itr.remote(itr=400) for sg in scene_gen_ray])
    dataset = jax.tree_map(lambda *x: np.concatenate(x, axis=0), *dataset)
    rb.push(dataset)
else:
    for i in range(rb.get_dir_size()):
        rb.load()
        if rb.get_size() >= data_limit:
            break

data_pnt = rb.sample(size=16, type='val')

# pcd = cutil.pcd_from_depth(data_pnt[0][1], data_pnt[0][3], pixel_size)
# import open3d as o3d
# for i in range(2):
#     pcd_pick = pcd[i]
#     oposquat = data_pnt[1][i]
#     oH = tutil.pq2H(oposquat[...,:3], oposquat[...,3:])
#     quat = get_quat_idx(tutil.Rm2q(oH[...,:3,:3]), projection=True)
#     oH_proj = oH
#     oH_proj = oH_proj.at[...,:3,:3].set(tutil.q2R(quat))
#     pcd_o3d = o3d.geometry.PointCloud()
#     pcd_o3d.points = o3d.utility.Vector3dVector(pcd_pick.reshape(-1,3))
#     mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
#         size=0.1, origin=[0, 0, 0])
#     mesh_frame_obj = o3d.geometry.TriangleMesh.create_coordinate_frame(
#         size=0.1, origin=[0, 0, 0])
#     mesh_frame_obj.transform(oH)
#     mesh_frame_obj_proj = o3d.geometry.TriangleMesh.create_coordinate_frame(
#         size=0.1, origin=[0, 0, 0])
#     mesh_frame_obj_proj.transform(oH_proj)
#     o3d.visualization.draw_geometries([pcd_o3d, mesh_frame,mesh_frame_obj, mesh_frame_obj_proj])

# plt.figure()
# for i in range(8):
#     plt.subplot(8,3,3*i+1)
#     plt.imshow(data_pnt[0][0][i])
#     plt.subplot(8,3,3*i+2)
#     plt.imshow(data_pnt[0][1][i])
#     plt.subplot(8,3,3*i+3)
#     plt.imshow(data_pnt[0][2][i])
# plt.show()


class Estimator(nn.Module):
    nsample = 200

    @nn.compact
    def __call__(self, rgb, depth, seg, intrinsic, jkey, quat=None, train=False):
        '''
        seg: instance pixel should be 1
        '''
        assert rgb.dtype == jnp.uint8
        if depth.shape[-1] != 1:
            depth = depth[...,None]
        if seg.shape[-1] != 1:
            seg = seg[...,None]

        if rgb.ndim==3:
            rgb, depth, seg, intrinsic = jax.tree_map(lambda x: x[None], (rgb, depth, seg, intrinsic))

        pcd_px = cutil.pcd_from_depth(depth, intrinsic, pixel_size)

        # flatten
        rgb_flat, seg_flat, pcd_flat = jax.tree_map(lambda x: einops.rearrange(x, '... i j k -> ... (i j) k'), (rgb, seg, pcd_px))

        # sample pixel index
        idx = jax.vmap(partial(jax.random.choice, a=rgb_flat.shape[-2], shape=(self.nsample,)))(jax.random.split(jkey, rgb.shape[0]), p=(seg_flat==1)[...,0].astype(jnp.float32))
        _, jkey = jax.random.split(jkey)

        rgb_sample, pcd_sample = jax.tree_map(lambda x: jnp.take_along_axis(x, idx[...,None], axis=-2), (rgb_flat, pcd_flat))

        # # visualization
        # import open3d as o3d
        # for i in range(rgb.shape[0]):
        #     pcd_pick = pcd_sample[i]
        #     oposquat = data_pnt[1][i]
        #     oH = tutil.pq2H(oposquat[...,:3], oposquat[...,3:])
        #     pcd_o3d = o3d.geometry.PointCloud()
        #     pcd_o3d.points = o3d.utility.Vector3dVector(pcd_pick.reshape(-1,3))
        #     mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        #         size=0.1, origin=[0, 0, 0])
        #     mesh_frame_obj = o3d.geometry.TriangleMesh.create_coordinate_frame(
        #         size=0.1, origin=[0, 0, 0])
        #     mesh_frame_obj.transform(oH)
        #     o3d.visualization.draw_geometries([pcd_o3d, mesh_frame,mesh_frame_obj])

        # normalize features
        rgb_sample = rgb_sample/255.
        pcd_mean = jnp.mean(pcd_sample, axis=-2, keepdims=True)
        pcd_sample = pcd_sample-pcd_mean

        # x = jnp.concatenate([pcd_sample, rgb_sample], axis=-1)
        x = pcd_sample

        # project point clouds
        for _ in range(2):
            x = nn.Dense(64)(x)
            x = nn.relu(x)
        x = jnp.concatenate([x, jnp.max(x, axis=-2, keepdims=True) + 0*x], axis=-1)
        for _ in range(2):
            x = nn.Dense(128)(x)
            x = nn.relu(x)
        x = jnp.max(x, axis=-2)
        quat_cat = nn.Dense(quat_nsample)(x)

        if quat is None:
            idx = jax.random.categorical(jkey, quat_cat)
            _, jkey = jax.random.split(jkey)
        else:
            idx = get_quat_idx(quat)
        quat = jnp.take_along_axis(quat_samples[None], idx[...,None,None], axis=-2)
        quat = jnp.squeeze(quat, axis=-2)

        quat_ft = nn.relu(nn.Dense(32)(quat))
        x_pos = jnp.concatenate([x, quat_ft], axis=-1)

        for _ in range(2):
            x_pos = nn.Dense(32)(x_pos)
            x_pos = nn.relu(x_pos)

        pos = nn.Dense(3)(x_pos)

        return pos + jnp.squeeze(pcd_mean, axis=-2), quat, quat_cat
        # return pos, quat, quat_cat



models = Estimator()
value, params = models.init_with_output(jkey, *data_pnt[0], jax.random.split(jkey)[1])
_, jkey = jax.random.split(jkey)

def loss_func(params_, datapnt_, jkey):
    x_, y_ = datapnt_
    pos_label, quat_label = y_[...,:3], y_[...,3:]
    pos_pred, _, quat_cat = models.apply(params_, *x_, jkey, quat=quat_label, train=True)

    idx = get_quat_idx(quat_label)

    # pos_loss_abs = jnp.sum(jnp.abs(pos_pred - pos_label), axis=-1)
    pos_loss = jnp.sum((pos_pred - pos_label)**2, axis=-1)

    quat_loss = -jnp.take_along_axis(nn.log_softmax(quat_cat, axis=-1), idx[...,None], axis=-1)
    quat_loss = jnp.squeeze(quat_loss, axis=-1)

    return jnp.mean(pos_loss) + jnp.mean(quat_loss), {'pos_los':jnp.mean(jnp.sqrt(pos_loss)), 'quat_loss':jnp.mean(quat_loss)}

# loss_func(params, *data_pnt)
loss_func_jit = jax.jit(loss_func)
loss_func_grad = jax.grad(loss_func, has_aux=True)
# loss_func_jit(params, *data_pnt)

# optimizer = optax.adam(1e-3)
optimizer = optax.adam(3e-4)
opt_state = optimizer.init(params)

def train_func(params_, opt_state, datapnt_, jkey):
    for i in range(inner_itr):
        _, jkey = jax.random.split(jkey)
        ridx = jax.random.permutation(jkey, datapnt_[1].shape[0])[:batch_size]
        datapnt_batch = jax.tree_map(lambda x: x[ridx], datapnt_)
        grad, metric = loss_func_grad(params_, datapnt_batch, jkey)
        updates, opt_state = optimizer.update(grad, opt_state, params_)
        params_ = optax.apply_updates(params_, updates)
    return params_, opt_state, metric, jkey

train_func_jit = jax.jit(train_func)
# train_func_jit = train_func

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(log_path ,mesh_name, current_time)

p.connect(p.DIRECT)
egl = pkgutil.get_loader('eglRenderer')
if (egl):
    pluginId = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
else:
    pluginId = p.loadPlugin("eglRendererPlugin")
print("pluginId=",pluginId)
gt_oid = p.loadURDF(os.path.join('objects',mesh_name,mesh_name+'.urdf'))
p.changeVisualShape(gt_oid, -1, textureUniqueId=-1, rgbaColor=(1,0,0,0.7))
pred_oid_list = []
for _ in range(5):
    pred_oid = p.loadURDF(os.path.join('objects',mesh_name,mesh_name+'.urdf'))
    p.changeVisualShape(pred_oid, -1, textureUniqueId=-1, rgbaColor=(0,0,1,0.2))
    pred_oid_list.append(pred_oid)
pm = p.computeProjectionMatrixFOV(fov=50, aspect=1, nearVal=0.001, farVal=5.0)
# vm = p.computeViewMatrix(cameraEyePosition=[0.0001,0,0], cameraTargetPosition=[0,0,-1], cameraUpVector=[0,1,0])
# vm = p.computeViewMatrix(cameraEyePosition=[0,1.,-0.5], cameraTargetPosition=[0,0,-1], cameraUpVector=[1,0,0])
def evaluation(params, dataset, itr, prefix, jkey):
    x,y = dataset
    pos_label, quat_label = y[...,:3], y[...,3:]
    # pos_pred, quat_pred, quat_cat = models.apply(params, *x, jkey)

    pos_pred_list = []
    quat_pred_list = []
    for _ in range(5):
        pos_pred_, quat_pred_, _ = models.apply(params, *x, jkey)
        _, jkey = jax.random.split(jkey)
        pos_pred_list.append(pos_pred_)
        quat_pred_list.append(quat_pred_)

    vis_rgb_list = []
    for ei in range(9):
        vm = p.computeViewMatrix(cameraEyePosition=pos_label[ei] + np.array([0,1,0]), cameraTargetPosition=pos_label[ei], cameraUpVector=[1,0,0])
        p.resetBasePositionAndOrientation(gt_oid, pos_label[ei], quat_label[ei])
        for i in range(5):
            p.resetBasePositionAndOrientation(pred_oid_list[i], pos_pred_list[i][ei], quat_pred_list[i][ei])
        cam_res = p.getCameraImage(width=200, height=200,
                        viewMatrix=vm, projectionMatrix=pm,
                        renderer=p.ER_BULLET_HARDWARE_OPENGL
                        )
        rgb = np.array(cam_res[2])[...,:3]
        vis_rgb_list.append(rgb)

    vis_rgb = np.array(vis_rgb_list)
    vis_rgb = einops.rearrange(vis_rgb, '(r1 r2) p1 p2 j -> (r1 p1) (r2 p2) j', r1=3, r2=3)
    target_dir = os.path.join(log_dir, 'img')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    Image.fromarray(vis_rgb).save(os.path.join(target_dir, str(itr)+prefix+'.png'))

evaluation(params, rb.sample(32, 'val'), 0, 'val', jkey)

for itr in range(1000000):
    # data generation
    if datagen:
        # scene_gen_ray = [ray.remote(sg.SceneGen).remote(pixel_size=pixel_size, meshname=mesh_name) for _ in range(env_no)]
        if itr%10 == 0 and datagen:
            dataset = ray.get([sg.gen_dataset_itr.remote(itr=100) for sg in scene_gen_ray])
            dataset = jax.tree_map(lambda *x: np.concatenate(x, axis=0), *dataset)
            rb.push(dataset)

    # if itr%200 == 0 and itr!= 0:
    #     rb.load()

    for i in range(10):
        datapnt = rb.sample(size=batch_size * inner_itr, type='train')
        _, jkey = jax.random.split(jkey)
        params, opt_state, train_metric, jkey = train_func_jit(params, opt_state, datapnt, jkey)

    if itr % 10 == 0:
        val_datapnt = rb.sample(size=batch_size, type='val')
        _, jkey = jax.random.split(jkey)
        _, val_metric = loss_func_jit(params, val_datapnt, jkey)
        train_metric = {'train/'+k:train_metric[k] for k in train_metric}
        val_metric = {'val/'+k:val_metric[k] for k in val_metric}
        metric = {**train_metric, **val_metric}
        _, jkey = jax.random.split(jkey)
        print(itr, metric)

    log_interval = 10
    save_interval = 100
    if itr % log_interval == 0:
        if itr % (5*log_interval) == 0:
            evaluation(params, rb.sample(32, 'val'), itr, 'val', jkey)

    if itr % save_interval == 0:
        save_dict = {'models':models, 'params':params, 'quat_samples':quat_samples}
        with open(os.path.join(log_dir, 'save_dict.pkl'), 'wb') as f:
            pickle.dump(save_dict, f)
        print('save params')