import pybullet as p
import numpy as np
import os
from scipy.spatial.transform import Rotation as sciR
import jax

import util.transform_util as tutil
import util.camera_util as cutil


def intrinsic_sample(pixel_size):
    # fov =np.random.uniform(35., 70.) # d455 / d435 / pixy 2.1
    fov =np.random.uniform(50., 70.) # d455 / d435 / pixy 2.1

    intrinsic = cutil.pbfov_to_intrinsic(pixel_size, fov)
    return intrinsic

class SceneGen(object):
    
    def __init__(self, gui=False, pixel_size=(100,100), meshname='obstacle2'):
        self.gui = gui
        self.pixel_size = pixel_size
        self.meshname = meshname
        if self.gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        self.init()

    def init(self):
        self.table_id = None
        self.objcls_list = []
        self.cambody_id = None
        p.setGravity(0,0,-9.81)

    
    def reset(self, reset=False):
        if reset:
            p.resetSimulation()
            self.init()

        # table
        # self.table_offset = np.random.uniform(-0.080, 0.080)
        self.table_offset = 0
        table_d = 1.0 # if set 0.03 -> cause egl render error : there is weird shadow in the floor
        if self.table_id is None:
            self.table_id = p.createMultiBody(baseMass=0.0, basePosition=[0,0,-table_d],
                                        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5,0.5,table_d]),
                                        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5,0.5,table_d]),
                                    )
        p.resetBasePositionAndOrientation(self.table_id, [0,0,-table_d+self.table_offset], [0,0,0,1]) # table height randomization

        # if len(self.objcls_list) != 0:
        #     for oc in self.objcls_list:
        #         p.removeBody(oc)
        #     self.objcls_list = []

        # self.cur_obj_no = np.random.randint(1, self.max_obj_no+1)
        self.cur_obj_no = 1
        # self.oposquat_list = []
        center_offset = np.random.uniform([-0.25,-0.25, 0.0], [0.25,0.25, 0], size=(3,))
        # while len(self.oposquat_list) < self.cur_obj_no:
        if len(self.objcls_list) == 0:
            oid = p.loadURDF(os.path.join('objects', self.meshname, self.meshname+'.urdf'))
            self.objcls_list.append(oid)
        else:
            oid = self.objcls_list[0]
        invalid_pose = True
        cnt = 0
        while invalid_pose:
            opos, oquat = center_offset+np.random.uniform([-0.1,-0.1, 0.0], [0.1,0.1,1.0], size=(3,)), tutil.qrand(outer_shape=())
            p.resetBasePositionAndOrientation(oid, opos, oquat)
            p.performCollisionDetection()
            if len(p.getContactPoints(oid)) == 0:
                invalid_pose = False
            cnt += 1
            if cnt >= 10:
                break
        if cnt >= 10:
            p.removeBody(oid)
        # self.oposquat_list.append((opos, oquat))

        # step simulation
        pep = p.getPhysicsEngineParameters()
        for ts in range(int(3.0/pep['fixedTimeStep'])):
            p.stepSimulation()
        self.contact_moments = [None]

        # get obj informations
        posquat_list = []
        for oc in self.objcls_list:
            posquat_list.append(np.concatenate(p.getBasePositionAndOrientation(oc), axis=-1))
        self.obj_posquat = np.stack(posquat_list, 0)

        # domain randomization
        p.changeVisualShape(self.table_id, -1, rgbaColor=list(np.random.uniform(0,1,size=(3,)))+[1])
        light_mag = np.random.uniform(8, 12)
        self.light_dir = light_mag*(sciR.from_euler('x', np.random.uniform(-np.pi/3, np.pi/3))*sciR.from_euler('z', np.random.uniform(-np.pi, np.pi))).as_matrix()[:,2]


    def get_rgb_from_camera_pdir(self, intrinsic, pos, quat):
        '''
        quat : z axis is aligned to opengl coordinate
        '''
        far = 3.0
        near = 0.001
        pm = p.computeProjectionMatrix(*cutil.intrinsic_to_pb_lrbt(intrinsic, near=near), nearVal=near, farVal=far)
        camH = np.zeros((4,4))
        camH[:3,:3] = sciR.from_quat(quat).as_matrix()
        camH[:3,3] = pos
        camH[3,3] = 1
        vm = np.linalg.inv(camH).T
        vm = vm.reshape(-1)

        cam_res = p.getCameraImage(width=int(intrinsic[0]), height=int(intrinsic[1]),
                        viewMatrix=vm.reshape(-1), projectionMatrix=pm, shadow=1,
                        lightDirection=self.light_dir,
                        renderer=p.ER_BULLET_HARDWARE_OPENGL
                        )
        rgb = np.array(cam_res[2])[...,:3]
        depth_buf = np.array(cam_res[3])
        seg = np.array(cam_res[4])
        if np.sum(seg==1) < 10:
            return None
        depth = far * near / (far - (far - near) * depth_buf)

        # pcd = cutil.pcd_from_depth_np(depth, intrinsic)

        # calculate labels
        opos, oquat = self.obj_posquat[0][:3], self.obj_posquat[0][3:]
        oH = np.zeros((4,4))
        oH[:3,:3] = sciR.from_quat(oquat).as_matrix()
        oH[:3,3] = opos
        oH[3,3] = 1

        oH_c = np.linalg.inv(camH)@oH

        pos_label = oH_c[:3,3]
        quat_label = sciR.from_matrix(oH_c[:3,:3]).as_quat()

        # import open3d as o3d
        # pcd_o3d = o3d.geometry.PointCloud()
        # pcd_o3d.points = o3d.utility.Vector3dVector(pcd.reshape(-1,3))
        # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        #     size=0.1, origin=[0, 0, 0])
        # mesh_frame_obj = o3d.geometry.TriangleMesh.create_coordinate_frame(
        #     size=0.1, origin=[0, 0, 0])
        # mesh_frame_obj.transform(oH_c)
        # o3d.visualization.draw_geometries([pcd_o3d, mesh_frame,mesh_frame_obj])

        return (np.array(rgb).astype(np.uint8), np.array(depth).astype(np.float16), 
                np.array(seg).astype(np.uint8), np.array(intrinsic).astype(np.float16)), np.concatenate([pos_label, quat_label], 0).astype(np.float16)

    def get_one_datapnt(self):
        self.reset()
        intrinsic = intrinsic_sample(self.pixel_size)

        cam_pos = np.array([np.random.uniform(0.4, 0.6), 0, 0.050])
        cam_pos += np.random.normal(scale=np.array([0.1,0.1,0.02]),size=(3,))
        cam_quat = tutil.line2q(zaxis=np.array([1,0,0]), yaxis=np.array([0,0,1]))

        return self.get_rgb_from_camera_pdir(intrinsic, cam_pos, cam_quat)


    def gen_dataset_itr(self, itr=10):
        data = []
        for _ in range(itr):
            dpnt = self.get_one_datapnt()
            if dpnt is not None:
                data.append(dpnt)

        return jax.tree_map(lambda *x: np.stack(x, 0), *data)


if __name__ == '__main__':
    # scene_cls = SceneGen(gui=True)

    # scene_cls.gen_dataset_itr(100)

    import ray
    import pickle
    import datetime
    pixel_size = (100,100)
    env_no = 10

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    meshname = 'hammer2'
    ds_dir = os.path.join('dataset', meshname)
    if not os.path.exists(ds_dir):
        os.mkdir(ds_dir)

    scene_gen_ray = [ray.remote(SceneGen).remote(pixel_size=pixel_size, meshname=meshname) for _ in range(env_no)]

    for i in range(100):
        dataset = ray.get([sg.gen_dataset_itr.remote(itr=1000) for sg in scene_gen_ray])
        dataset = jax.tree_map(lambda *x: np.concatenate(x, axis=0), *dataset)

        with open(os.path.join(ds_dir,f'{current_time}_{i}.pkl'), 'wb')as f:
            pickle.dump(dataset, f)
        print(f'save datapoints {i}')

    