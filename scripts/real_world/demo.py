import argparse
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import pickle
import pybullet as pb
from scipy.spatial.transform import Rotation
import rospy
import threading
import time
from typing import Any, Dict, Optional, Tuple

from online_isec import constants
from online_isec import perception
from online_isec.point_cloud_proxy_sync import RealsenseStructurePointCloudProxy
from src.real_world.ur5 import UR5
import online_isec.utils as isec_utils
from pybullet_planning.pybullet_tools import utils as pu
import utils


def worker(ur5: UR5, sphere: int, mug: int, data: Dict[str, Any]):

    while True:

        if data["stop"]:
            break

        gripper_pos, gripper_quat = ur5.get_end_effector_pose()
        gripper_pos = gripper_pos - constants.DESK_CENTER

        if data["T"] is not None:
            # Make the mug follow the gripper.
            T_g_to_b = utils.pos_quat_to_transform(gripper_pos, gripper_quat)
            T_m_to_b = np.matmul(T_g_to_b, data["T"])
            m_pos, m_quat = utils.transform_to_pos_quat(T_m_to_b)
            pu.set_pose(mug, (m_pos, m_quat))

        # Mark the gripper with a sphere.
        pu.set_position(sphere, gripper_pos[0], gripper_pos[1], gripper_pos[2])
        time.sleep(0.1)


def get_knn_and_deltas(obj: NDArray, vps: NDArray, k: int=10) -> Tuple[NDArray, NDArray]:

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(obj[:, 0], obj[:, 1], obj[:, 2], color="red", alpha=0.1)
    # ax.scatter(vps[:, 0], vps[:, 1], vps[:, 2], color="green")
    # plt.show()

    dists = np.sum(np.square(obj[None] - vps[:, None]), axis=-1)
    knn_list = []
    deltas_list = []

    for i in range(dists.shape[0]):
        # Get K closest points, compute relative vectors.
        knn = np.argpartition(dists[i], k)[:k]
        deltas = vps[i: i + 1] - obj[knn]
        knn_list.append(knn)
        deltas_list.append(deltas)

    knn_list = np.stack(knn_list)
    deltas_list = np.stack(deltas_list)
    return knn_list, deltas_list


def save_pick_pose(observed_pc: NDArray[np.float32], canon_mug: Dict[str, Any],
                   mug_param: Tuple[NDArray, NDArray, NDArray], ur5: UR5, save_path: Optional[bool]=None):

    T_g_to_b = utils.pos_quat_to_transform(*ur5.get_end_effector_pose())
    T_m_to_b = utils.pos_rot_to_transform(mug_param[1] + constants.DESK_CENTER, utils.yaw_to_rot(mug_param[2]))
    T_g_to_m = np.matmul(np.linalg.inv(T_m_to_b), T_g_to_b)

    # Grasp in a canonical frame.
    g_to_m_pos, g_to_m_quat = utils.transform_to_pos_quat(T_g_to_m)

    # Warped object in a canonical frame.
    mug_pc_complete = utils.canon_to_pc(canon_mug, mug_param)

    # Index of the point on the canonical object closest to a point between the gripper fingers.
    dist = np.sqrt(np.sum(np.square(mug_pc_complete - g_to_m_pos), axis=1))
    index = np.argmin(dist)

    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump({
                "index": index,
                "pos": g_to_m_pos,
                "quat": g_to_m_quat,
                "T_g_to_b": T_g_to_b,
                "observed_pc": observed_pc,
            }, f)


def save_place_contact_points(
    tree_pc: NDArray, ur5: UR5, mug: int, tree: int, T_m_to_g: NDArray, T_g_pre_to_b: NDArray, canon_mug: Dict[str, Any], canon_tree: Dict[str, Any],
    mug_param: Tuple[NDArray, NDArray, NDArray], tree_param: Tuple[NDArray, NDArray], delta: float=0.1,
    save_path: Optional[str]=None):

    # Find close points between the mug and the tree in pybullet.
    pb.performCollisionDetection()
    cols = pb.getClosestPoints(mug, tree, delta)

    spheres_mug = []
    spheres_tree = []
    for col in cols:
        pos_mug = col[5]
        with pu.HideOutput():
            s = pu.load_model("../data/sphere_red.urdf")
            pu.set_pose(s, pu.Pose(pu.Point(*pos_mug)))
        spheres_mug.append(s)

        pos_tree = col[6]
        with pu.HideOutput():
            s = pu.load_model("../data/sphere.urdf")
            pu.set_pose(s, pu.Pose(pu.Point(*pos_tree)))
        spheres_tree.append(s)

    pos_mug = [col[5] for col in cols]
    pos_tree = [col[6] for col in cols]

    pos_mug = np.stack(pos_mug, axis=0).astype(np.float32)
    pos_tree = np.stack(pos_tree, axis=0).astype(np.float32)

    T_g_to_b = utils.pos_quat_to_transform(*ur5.get_end_effector_pose())
    T_g_pre_to_g = np.matmul(np.linalg.inv(T_g_to_b), T_g_pre_to_b)
    T_m_to_b = np.matmul(T_g_to_b, T_m_to_g)
    T_ws_to_b = isec_utils.workspace_to_base()
    T_m_to_ws = np.matmul(np.linalg.inv(T_ws_to_b), T_m_to_b)

    T_t_to_ws = utils.pos_rot_to_transform(tree_param[0], utils.yaw_to_rot(tree_param[1]))

    mug_pc_origin = utils.canon_to_pc(canon_mug, mug_param)
    pos_tree_tree_coords = utils.transform_pointcloud_2(pos_tree, np.linalg.inv(T_t_to_ws))
    pos_tree_mug_coords = utils.transform_pointcloud_2(pos_tree, np.linalg.inv(T_m_to_ws))

    knns, deltas = get_knn_and_deltas(mug_pc_origin, pos_tree_mug_coords)

    dist_2 = np.sqrt(np.sum(np.square(canon_tree["canonical_obj"][:, None] - pos_tree_tree_coords[None]), axis=2))  # TODO: is this correct?
    i_2 = np.argmin(dist_2, axis=0).transpose()

    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump({
                "knns": knns,
                "deltas": deltas,
                "target_indices": i_2,
                "T_g_to_b": T_g_to_b,
                "T_g_pre_to_g": T_g_pre_to_g,
                "observed_pc": tree_pc,
            }, f)


def main(args):

    rospy.init_node("easy_perception")
    pc_proxy = RealsenseStructurePointCloudProxy()

    ur5 = UR5(setup_planning=True)
    # ur5.plan_and_execute_joints_target(ur5.home_joint_values)
    ur5.gripper.open_gripper(position=70)

    mug_pc_complete, mug_param, tree_pc_complete, tree_param, canon_mug, canon_tree, mug_pc, tree_pc = perception.mug_tree_perception(
        pc_proxy, np.array(constants.DESK_CENTER), ur5.tf_proxy, ur5.moveit_scene,
        add_mug_to_planning_scene=True, add_tree_to_planning_scene=True, rviz_pub=ur5.rviz_pub,
        mug_save_decomposition=False,
    )

    # Setup simulation with what we see in the real world.
    pu.connect(use_gui=True, show_sliders=False)
    pu.set_default_camera(distance=2)
    pu.disable_real_time()
    pu.draw_global_system()

    mug = pu.load_model("../tmp.urdf", pu.Pose(mug_param[1], pu.Euler(yaw=mug_param[2])))
    tree = pu.load_model("../data/real_tree.urdf", pu.Pose(tree_param[0], pu.Euler(yaw=tree_param[1])), fixed_base=True)
    point = pu.load_model("../data/sphere.urdf")

    # Continuously update the gripper position in simulation.
    data = {
        "T": None,
        "stop": False,
    }
    thread = threading.Thread(target=worker, args=(ur5, point, mug, data))
    thread.start()

    input("Close gripper?")
    ur5.gripper.close_gripper()
    save_pick_pose(mug_pc, canon_mug, mug_param, ur5, args.pick_save_path + ".pkl")

    # Calculate mug to gripper transform. Transmit it to simulation.
    T_g_to_b = utils.pos_quat_to_transform(*ur5.get_end_effector_pose())
    T_ws_to_b = isec_utils.workspace_to_base()
    T_g_to_ws = np.matmul(np.linalg.inv(T_ws_to_b), T_g_to_b)

    T_m_to_ws = utils.pos_rot_to_transform(mug_param[1], utils.yaw_to_rot(mug_param[2]))
    T_m_to_g = np.matmul(np.linalg.inv(T_g_to_ws), T_m_to_ws)
    data["T"] = T_m_to_g

    # Save any number of waypoints.
    input("Save place waypoint?")
    T_g_pre_to_b = utils.pos_quat_to_transform(*ur5.get_end_effector_pose())

    input("Save place pose?")
    save_place_contact_points(
        tree_pc, ur5, mug, tree, T_m_to_g, T_g_pre_to_b, canon_mug, canon_tree, mug_param, tree_param, save_path=args.place_save_path + ".pkl")

    # Reset.
    # input("Open gripper?")
    ur5.gripper.open_gripper()
    # ur5.plan_and_execute_joints_target(ur5.home_joint_values)

    data["stop"] = True
    thread.join()


parser = argparse.ArgumentParser()
parser.add_argument("--show-perception", default=False, action="store_true")
parser.add_argument("--pick-save-path", type=str, default="data/230201_real_pick_clone", help="Postfix added automatically.")
parser.add_argument("--place-save-path", type=str, default="data/230201_real_place_clone", help="Postfix added automatically.")
main(parser.parse_args())
