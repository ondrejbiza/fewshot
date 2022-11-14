import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import pybullet as pb

from pybullet_planning.pybullet_tools import utils as pu
from pybullet_planning.pybullet_tools.ikfast.franka_panda.ik import PANDA_INFO, FRANKA_URDF
import utils
import viz_utils

WORKSPACE_LOW = np.array([0.25, -0.5, 0.], dtype=np.float32)
# TODO: what to do with the z-axis?
WORKSPACE_HIGH = np.array([0.75, 0.5, 0.28], dtype=np.float32)


def setup_scene(mug_index, tree_index):

    # setup simulated scene
    pu.connect(use_gui=True, show_sliders=True)
    pu.set_default_camera(distance=2)
    pu.disable_real_time()
    pu.draw_global_system()

    with pu.HideOutput():
        floor = pu.load_model("models/short_floor.urdf", fixed_base=True)
        robot = pu.load_model(FRANKA_URDF, fixed_base=True)

        placed = []

        mug = pu.load_model("../data/mugs/test/{:d}.urdf".format(mug_index))
        utils.place_object(mug, floor, placed, WORKSPACE_LOW, WORKSPACE_HIGH)
        placed.append(mug)

        tree = pu.load_model("../data/simple_trees/test/{:d}.urdf".format(tree_index), fixed_base=True)
        utils.place_object(tree, floor, placed, WORKSPACE_LOW, WORKSPACE_HIGH)
        placed.append(tree)

    return mug, tree


def observe_and_fill_in(mug, tree):

    # get a point cloud
    pcs, _ = utils.observe_point_cloud(utils.RealSenseD415.CONFIG, [mug, tree])

    # load canonical objects
    canon = {}
    with open("data/mugs_pca.pkl", "rb") as f:
        canon[mug] = pickle.load(f)
    with open("data/simple_trees_pca.pkl", "rb") as f:
        canon[tree] = pickle.load(f)

    # fit canonical objects to observed point clouds
    new_obj_1, _, param_1 = utils.planar_pose_warp_gd(canon[mug]["pca"], canon[mug]["canonical_obj"], pcs[mug])
    new_obj_2, _, param_2 = utils.planar_pose_warp_gd(canon[tree]["pca"], canon[tree]["canonical_obj"], pcs[tree], object_size_reg=0.1)
    viz_utils.show_scene({mug: new_obj_1, tree: new_obj_2}, background=np.concatenate(list(pcs.values())))
    return new_obj_1, param_1, new_obj_2, param_2, canon


def main(args):

    mug, tree = setup_scene(args.mug_index, args.tree_index)
    new_obj_1, param_1, new_obj_2, param_2, canon = observe_and_fill_in(mug, tree)

    # get registered points on the warped canonical object
    with open(args.load_path, "rb") as f:
        place_data = pickle.load(f)

    knns = place_data["knns"]
    deltas = place_data["deltas"]
    target_indices = place_data["target_indices"]

    # bring delta into the reference frame of the current mug
    # !!! only rotate, do not translate
    rot = utils.yaw_to_rot(param_1[2])
    new_deltas = np.matmul(deltas, rot.T)

    anchors = new_obj_1[knns]
    targets = np.mean(anchors + new_deltas, axis=1)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(new_obj_1[:, 0], new_obj_1[:, 1], new_obj_1[:, 2], color="red", alpha=0.1)
    # ax.scatter(targets[:, 0], targets[:, 1], targets[:, 2], color="green")
    # plt.show()

    points_2 = new_obj_2[target_indices]
    print(points_2.shape)

    # find best pair-wise fit
    print(targets.shape, points_2.shape)
    vp_to_p2, _, _ = utils.best_fit_transform(targets, points_2)
    print("Best fit spatial transform:")
    print(vp_to_p2)

    # create a spatial transformation matrix for the mug
    mug_pos, mug_rot = pu.get_pose(mug)
    mug_pos = np.array(mug_pos, dtype=np.float32)
    mug_rot = pu.matrix_from_quat(mug_rot)
    
    mug_T = np.concatenate([mug_rot, mug_pos[:, None]], axis=1)
    mug_T = np.concatenate([mug_T, np.array([[0., 0., 0., 1.]])], axis=0)
    print("Original mug spatial transform:")
    print(mug_T)

    # get a new mug pose
    new_mug_T = np.matmul(vp_to_p2, mug_T)
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

    pu.wait_if_gui()
    save = pu.WorldSaver()
    pos, quat = utils.wiggle(mug, tree)
    save.restore()
    pu.set_pose(mug, (pos, quat))

    pb.performCollisionDetection()
    print("In collision:", pu.body_collision(mug, tree))

    pu.wait_if_gui()

    for i in range(100):
        pu.enable_gravity()
        pu.step_simulation()

    pu.wait_if_gui()

    pu.disconnect()


parser = argparse.ArgumentParser()
parser.add_argument("load_path")
parser.add_argument("-m", "--mug-index", type=int, default=0)
parser.add_argument("-t", "--tree-index", type=int, default=0)
main(parser.parse_args())
