import argparse
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
        floor = pu.load_pybullet("pybullet_planning/models/short_floor.urdf", fixed_base=True)  # TODO: fixed base
        mug = pu.load_pybullet("../data/mugs/test/{:d}.urdf".format(args.mug_index))
        tree = pu.load_pybullet("../data/simple_trees/test/{:d}.urdf".format(args.tree_index), fixed_base=True)  # TODO: fixed base

    pu.set_pose(mug, pu.Pose(pu.Point(x=0.2, y=0.0, z=pu.stable_z(mug, floor)), pu.Euler(0., 0., 0.)))
    pu.set_pose(tree, pu.Pose(pu.Point(x=0.0, y=0.0, z=pu.stable_z(tree, floor))))

    # get a point cloud
    pcs, _ = utils.observe_point_cloud(utils.RealSenseD415.CONFIG, [1, 2])

    # load canonical objects
    canon = {}
    with open("data/mugs_pca.pkl", "rb") as f:
        canon[1] = pickle.load(f)
    with open("data/simple_trees_pca.pkl", "rb") as f:
        canon[2] = pickle.load(f)

    # fit canonical objects to observed point clouds
    new_obj_1, _, _ = utils.planar_pose_warp_gd(canon[1]["pca"], canon[1]["canonical_obj"], pcs[1])
    new_obj_2, _, _ = utils.planar_pose_warp_gd(canon[2]["pca"], canon[2]["canonical_obj"], pcs[2], n_angles=1, object_size_reg=0.1)

    # get registered points on the warped canonical object
    indices = np.load(args.load_path)
    print("Indices:", indices)

    points_1 = new_obj_1[indices[0]]
    points_2 = new_obj_2[indices[1]]

    # find best pair-wise fit
    p1_to_p2, _, _ = utils.best_fit_transform(points_1, points_2)
    print("Best fit spatial transform:")
    print(p1_to_p2)

    # create a spatial transformation matrix for the mug
    mug_pos, mug_rot = pu.get_pose(mug)
    mug_pos = np.array(mug_pos, dtype=np.float32)
    mug_rot = pu.matrix_from_quat(mug_rot)
    
    mug_T = np.concatenate([mug_rot, mug_pos[:, None]], axis=1)
    mug_T = np.concatenate([mug_T, np.array([[0., 0., 0., 1.]])], axis=0)
    print("Original mug spatial transform:")
    print(mug_T)

    # get a new mug pose
    new_mug_T = np.matmul(mug_T, p1_to_p2)
    print("New mug spatial transform:")
    print(new_mug_T)

    # make pose pybullet compatible
    new_mug_rot = new_mug_T[:3, :3]
    new_mug_pos = new_mug_T[:3, 3]
    new_mug_quat= pu.quat_from_matrix(new_mug_rot)

    pu.set_pose(mug, (new_mug_pos, new_mug_quat))

    # get collisions
    pb.performCollisionDetection()
    print("In collision:", pu.body_collision(mug, tree))

    save = pu.WorldSaver()
    pos, quat = utils.wiggle(mug, tree)
    save.restore()
    pu.set_pose(mug, (pos, quat))

    pb.performCollisionDetection()
    print("In collision:", pu.body_collision(mug, tree))

    pu.wait_if_gui()
    pu.disconnect()


parser = argparse.ArgumentParser()
parser.add_argument("load_path")
parser.add_argument("-m", "--mug-index", type=int, default=0)
parser.add_argument("-t", "--tree-index", type=int, default=0)
main(parser.parse_args())
