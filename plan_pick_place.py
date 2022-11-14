import argparse
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
import pybullet as pb

from pybullet_planning.pybullet_tools.ikfast.franka_panda.ik import PANDA_INFO, FRANKA_URDF
from pybullet_planning.pybullet_tools import utils as pu
from robot import Panda, Robot
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

        mug = pu.load_model("../data/mugs/test/{:d}.urdf".format(mug_index))
        pu.set_pose(mug, pu.Pose(pu.Point(x=0.5, y=-0.25, z=pu.stable_z(mug, floor))))

        tree = pu.load_model("../data/simple_trees/test/{:d}.urdf".format(tree_index), fixed_base=True)
        pu.set_pose(tree, pu.Pose(pu.Point(x=0.5, y=0.25, z=pu.stable_z(tree, floor))))

    return mug, tree, robot, floor


def pick(mug: int, tree: int, robot: Robot, floor: int, pick_path: str):

    objects = [mug, tree, floor]

    # get registered points on the warped canonical object
    pick_indices = np.load(pick_path)

    # get a point cloud
    pcs, _ = utils.observe_point_cloud(utils.RealSenseD415.CONFIG, [mug, tree])

    # load canonical objects
    canon = {}
    with open("data/mugs_pca.pkl", "rb") as f:
        canon[mug] = pickle.load(f)
    with open("data/simple_trees_pca.pkl", "rb") as f:
        canon[tree] = pickle.load(f)

    # fit canonical objects to observed point clouds
    new_obj_1, _, _ = utils.planar_pose_warp_gd(canon[mug]["pca"], canon[mug]["canonical_obj"], pcs[mug])
    new_obj_2, _, _ = utils.planar_pose_warp_gd(canon[tree]["pca"], canon[tree]["canonical_obj"], pcs[tree], object_size_reg=0.1)
    filled_pcs = {mug: new_obj_1, tree: new_obj_2}
    # viz_utils.show_scene(filled_pcs, background=np.concatenate(list(pcs.values())))

    # get grasp
    position = new_obj_1[pick_indices]
    print("Position:", position)

    pose = pu.Pose(position, pu.Euler(roll=np.pi))

    # pu.wait_if_gui()

    path = robot.plan_grasp(pose, 0.15, objects)
    robot.execute_path(path)
    robot.open_hand()

    # pu.wait_if_gui()

    # path = robot.plan_grasp(pose, 0.1, [mug])
    path = robot.plan_grasp_naive(pose, 0.09)
    robot.execute_path(path)
    robot.close_hand()

    print("Picked object:", robot.get_picked_object(objects))

    # pu.wait_if_gui()

    mug_pos, mug_rot = pu.get_pose(mug)
    mug_pos = np.array(mug_pos, dtype=np.float32)
    mug_rot = pu.matrix_from_quat(mug_rot)
    
    mug_T = np.concatenate([mug_rot, mug_pos[:, None]], axis=1)
    mug_T = np.concatenate([mug_T, np.array([[0., 0., 0., 1.]])], axis=0)

    hand_T = pu.get_link_pose(robot.robot, robot.tool_link)
    hand_T = np.concatenate([pu.matrix_from_quat(hand_T[1]), np.array(hand_T[0], dtype=np.float32)[:, None]], axis=1)
    hand_T = np.concatenate([hand_T, np.array([[0., 0., 0., 1.]])], axis=0)

    g_t_m = np.matmul(np.linalg.inv(hand_T), mug_T)
    g_t_m_pos = g_t_m[:3, 3]
    g_t_m_quat = pu.quat_from_matrix(g_t_m[:3, :3])

    mug_attach = pu.Attachment(robot.robot, robot.tool_link, (g_t_m_pos, g_t_m_quat), mug)

    pose = robot.init_tool_pos, pu.get_link_pose(robot.robot, robot.tool_link)[1]
    path = robot.plan_motion(pose, [tree, floor], attachments=[mug_attach])
    robot.execute_path(path)

    print("Picked object:", robot.get_picked_object(objects))


def place(mug: int, tree: int, robot: Robot, floor: int, place_path: str):

    with open(place_path, "rb") as f:
        place_data = pickle.load(f)

    knns = place_data["knns"]
    deltas = place_data["deltas"]
    target_indices = place_data["target_indices"]

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
    
    # bring delta into the reference frame of the current mug
    # !!! only rotate, do not translate
    rot = utils.yaw_to_rot(param_1[2])
    new_deltas = np.matmul(deltas, rot.T)

    anchors = new_obj_1[knns]
    targets = np.mean(anchors + new_deltas, axis=1)

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
    # new_mug_T = np.matmul(T, mug_T)
    # print("New mug spatial transform:")
    # print(new_mug_T)

    hand_T = pu.get_link_pose(robot.robot, robot.tool_link)
    hand_T = np.concatenate([pu.matrix_from_quat(hand_T[1]), np.array(hand_T[0], dtype=np.float32)[:, None]], axis=1)
    hand_T = np.concatenate([hand_T, np.array([[0., 0., 0., 1.]])], axis=0)

    # w_t_g^-1 w_t_m
    g_t_m = np.matmul(np.linalg.inv(hand_T), mug_T)
    print("g_t_m:", g_t_m)
    g_t_m_pos = g_t_m[:3, 3]
    g_t_m_quat = pu.quat_from_matrix(g_t_m[:3, :3])

    new_hand_T = np.matmul(vp_to_p2, hand_T)
    print("new_hand_T:", new_hand_T)

    # make pose pybullet compatible
    new_hand_rot = new_hand_T[:3, :3]
    new_hand_pos = new_hand_T[:3, 3]
    new_hand_quat = pu.quat_from_matrix(new_hand_rot)

    new_mug_T = np.matmul(vp_to_p2, mug_T)
    new_mug_rot = new_mug_T[:3, :3]
    new_mug_pos = new_mug_T[:3, 3]
    new_mug_quat = pu.quat_from_matrix(new_mug_rot)

    save = pu.WorldSaver()
    pu.set_pose(mug, (new_mug_pos, new_mug_quat))
    pose = utils.wiggle(mug, tree)
    pu.set_pose(mug, pose)
    print("In collision:", pu.body_collision(mug, tree))
    # pu.wait_if_gui()
    save.restore()

    mug_attach = pu.Attachment(robot.robot, robot.tool_link, (g_t_m_pos, g_t_m_quat), mug)

    pose = (new_hand_pos, new_hand_quat)

    # conf = robot.plan_motion(pose, obstacles=[tree], attachments=[mug_attach])
    # save = pu.WorldSaver()
    # conf = robot.ik(pose)
    # pu.set_joint_positions(robot.robot, robot.ik_joints, conf)
    # mug_attach.assign()
    # pu.wait_if_gui()
    # save.restore()

    # print("In collision:", pu.body_collision(mug, tree))

    # pu.step_simulation()
    # cols = pb.getContactPoints(mug, tree)
    # print("Num colissions:", len(cols))

    # import time ; time.sleep(100)

    path = robot.plan_motion(pose, obstacles=[tree, floor], attachments=[mug_attach])
    robot.execute_path(path)
    robot.open_hand()

    # pu.set_pose(mug, (new_mug_pos, new_mug_quat))


def main(args):

    mug, tree, robot_id, floor = setup_scene(args.mug_index, args.tree_index)
    robot = Panda(robot=robot_id)

    pick(mug, tree, robot, floor, args.pick_path)
    pu.wait_if_gui()

    place(mug, tree, robot, floor, args.place_path)
    pu.wait_if_gui()

    pu.disconnect()


parser = argparse.ArgumentParser()
parser.add_argument("--pick-path", default="data/clone_pick.npy")
parser.add_argument("--place-path", default="data/clone_place.pickle")
parser.add_argument("-m", "--mug-index", type=int, default=0)
parser.add_argument("-t", "--tree-index", type=int, default=0)
main(parser.parse_args())
