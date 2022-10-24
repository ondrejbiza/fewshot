import argparse
import time
import os
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
import pybullet as pb
from pybullet_planning.pybullet_tools import utils as pu
import utils


def main(args):

    # setup simulated scene
    pu.connect(use_gui=True, show_sliders=True)
    pu.set_default_camera(distance=2)
    pu.disable_real_time()
    pu.draw_global_system()

    with pu.HideOutput():
        floor = pu.load_model('models/short_floor.urdf')
        mug = pu.load_model("../data/mugs/test/0.urdf")
        tree = pu.load_model("../data/trees/test/0.urdf")

    pu.set_pose(mug, pu.Pose(pu.Point(x=0.2, y=0.0, z=pu.stable_z(mug, floor)), pu.Euler(0., 0., 0.)))
    pu.set_pose(tree, pu.Pose(pu.Point(x=0.0, y=0.0, z=pu.stable_z(tree, floor))))

    # get a point cloud
    pcs, _ = utils.observe_point_cloud(utils.RealSenseD415.CONFIG, [1, 2])

    # load canonical objects
    canon = {}
    with open("data/mugs_pca.pkl", "rb") as f:
        canon[1] = pickle.load(f)
    with open("data/trees_pca_8d.pkl", "rb") as f:
        canon[2] = pickle.load(f)

    # fit canonical objects to observed point clouds
    new_obj_1, _, p1 = utils.planar_pose_warp_gd(canon[1]["pca"], canon[1]["canonical_obj"], pcs[1])
    new_obj_2, _, p2 = utils.planar_pose_warp_gd(canon[2]["pca"], canon[2]["canonical_obj"], pcs[2], n_angles=1)

    latent1, pos1, ang1 = p1
    latent2, pos2, ang2 = p2

    rot1 = utils.yaw_to_rot(ang1)
    rot2 = utils.yaw_to_rot(ang2)

    indices = np.load(args.load_path)

    # latent_1 = interpolate_pca(canon[1]["pca"], canon[1]["canonical_obj"], indices[0])
    # print("latent 1:", latent_1)
    # latent_2 = interpolate_pca(canon[2]["pca"], canon[2]["canonical_obj"], indices[1])
    # print("latent 2:", latent_2)

    # new_obj_1 = canon[1]["canonical_obj"] + canon[1]["pca"].inverse_transform(np.array([latent1])).reshape((-1, 3)) / 2.
    # new_obj_2 = canon[2]["canonical_obj"] + canon[2]["pca"].inverse_transform(np.array([latent2])).reshape((-1, 3)) / 2. 

    points_1 = new_obj_1[indices[0]]
    points_2 = new_obj_2[indices[1]]

    T, R, t = utils.best_fit_transform(points_1, points_2)

    tmp_pos, tmp_rot = pu.get_pose(mug)
    tmp_pos = np.array(tmp_pos, dtype=np.float32)
    tmp_rot = np.array(pb.getMatrixFromQuaternion(tmp_rot), dtype=np.float32).reshape((3, 3))
    
    tmp_T = np.concatenate([tmp_rot, tmp_pos[:, None]], axis=1)
    tmp_T = np.concatenate([tmp_T, np.array([[0., 0., 0., 1.]])], axis=0)

    print("T", T)
    print("tmp_T", tmp_T)

    tmp_T = np.matmul(T, tmp_T)

    print("tmp_T", tmp_T)

    tmp_rot = tmp_T[:3, :3]
    tmp_pos = tmp_T[:3, 3]

    print(tmp_rot.shape, tmp_pos.shape)

    r = Rotation.from_matrix(tmp_rot).as_quat()

    pu.set_pose(mug, (tmp_pos, r))

    time.sleep(100)

    pu.disconnect()


parser = argparse.ArgumentParser()
parser.add_argument("load_path")
main(parser.parse_args())
