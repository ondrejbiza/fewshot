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
from typing import Any, Dict

from online_isec import constants
from online_isec import perception
from online_isec.point_cloud_proxy import RealsenseStructurePointCloudProxy
from online_isec.ur5 import UR5
from pybullet_planning.pybullet_tools import utils as pu
import utils


def worker(ur5: UR5, sphere: int, mug: int, data: Dict[str, Any]):

    while True:

        if data["stop"]:
            break

        gripper_pos, gripper_quat = ur5.get_end_effector_pose()
        gripper_pos = gripper_pos - constants.DESK_CENTER

        if data["T"] is not None:
            T_g_to_b = utils.pos_quat_to_transform(gripper_pos, gripper_quat)
            T_m_to_b = np.matmul(T_g_to_b, data["T"])
            m_pos, m_quat = utils.transform_to_pos_quat(T_m_to_b)
            pu.set_pose(mug, (m_pos, m_quat))

        pu.set_position(sphere, gripper_pos[0], gripper_pos[1], gripper_pos[2])
        time.sleep(0.1)


def get_knn_and_deltas(obj, vps):

    k = 10
    # [n_pairs, n_points, 3]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(obj[:, 0], obj[:, 1], obj[:, 2], color="red", alpha=0.1)
    ax.scatter(vps[:, 0], vps[:, 1], vps[:, 2], color="green")
    plt.show()

    dists = np.sum(np.square(obj[None] - vps[:, None]), axis=-1)
    knn_list = []
    deltas_list = []

    for i in range(dists.shape[0]):
        knn = np.argpartition(dists[i], k)[:k]
        deltas = vps[i: i + 1] - obj[knn]
        knn_list.append(knn)
        deltas_list.append(deltas)

    knn_list = np.stack(knn_list)
    deltas_list = np.stack(deltas_list)
    return knn_list, deltas_list


def save_pick_pose(filled_and_transformed_mug: NDArray[np.float32], gripper_pos: NDArray[np.float32],
                   gripper_rot: NDArray[np.float32], save: bool):

    dist = np.sqrt(np.sum(np.square(filled_and_transformed_mug - gripper_pos), axis=1))
    index = np.argmin(dist)

    if save:
        with open("data/real_pick_clone.pkl", "wb") as f:
            pickle.dump({
                "index": index,
                "quat": gripper_rot
            }, f)


def save_place_contact_points(ur5, mug, tree, T_m_to_g, canon_mug, canon_tree, mug_param, tree_param, save: bool):

    pb.performCollisionDetection()
    cols = pb.getClosestPoints(mug, tree, 0.01)

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

    gripper_pos, gripper_quat = ur5.get_end_effector_pose()
    T_g_to_b = utils.pos_quat_to_transform(gripper_pos - constants.DESK_CENTER, gripper_quat)
    T_m_to_b = np.matmul(T_g_to_b, T_m_to_g)

    T_t_to_b = utils.pos_rot_to_transform(tree_param[0], utils.yaw_to_rot(tree_param[1]))

    mug_pc_origin = utils.canon_to_pc(canon_mug, mug_param)
    pos_tree_tree_coords = utils.transform_pointcloud_2(pos_tree, np.linalg.inv(T_t_to_b))  # TODO: is this correct?
    pos_tree_mug_coords = utils.transform_pointcloud_2(pos_tree, np.linalg.inv(T_m_to_b))  # TODO: is this correct?

    knns, deltas = get_knn_and_deltas(mug_pc_origin, pos_tree_mug_coords)  # TODO: is this correct?

    dist_2 = np.sqrt(np.sum(np.square(canon_tree["canonical_obj"][:, None] - pos_tree_tree_coords[None]), axis=2))  # TODO: is this correct?
    i_2 = np.argmin(dist_2, axis=0).transpose()

    if save:
        with open("data/real_place_clone.pkl", "wb") as f:
            pickle.dump({
                "knns": knns,
                "deltas": deltas,
                "target_indices": i_2
            }, f)


def main(args):

    rospy.init_node("easy_perception")
    pc_proxy = RealsenseStructurePointCloudProxy()

    ur5 = UR5(setup_planning=True)
    # ur5.plan_and_execute_joints_target(ur5.home_joint_values)

    mug_pc_complete, mug_param, tree_pc_complete, tree_param, canon_mug, canon_tree = perception.mug_tree_perception(
        pc_proxy, np.array(constants.DESK_CENTER),
        add_mug_to_planning_scene=False, add_tree_to_planning_scene=False
    )

    pu.connect(use_gui=True, show_sliders=False)
    pu.set_default_camera(distance=2)
    pu.disable_real_time()
    pu.draw_global_system()

    mug = pu.load_model("../tmp.urdf", pu.Pose(mug_param[1], pu.Euler(yaw=mug_param[2])))
    tree = pu.load_model("../data/real_tree.urdf", pu.Pose(tree_param[0], pu.Euler(yaw=tree_param[1])), fixed_base=True)

    point = pu.load_model("../data/sphere.urdf")
    data = {
        "T": None,
        "stop": False,
    }

    thread = threading.Thread(target=worker, args=(ur5, point, mug, data))
    thread.start()
    input("Close gripper?")

    gripper_pos, gripper_rot = ur5.get_end_effector_pose()
    gripper_pos = gripper_pos - constants.DESK_CENTER

    # TODO: We do not account for the mug moving as it is picked.
    tmp = np.matmul(
        utils.yaw_to_rot(mug_param[2]).T,
        Rotation.from_quat(gripper_rot).as_matrix()
    )
    gripper_rot = Rotation.from_matrix(tmp).as_quat()

    save_pick_pose(mug_pc_complete, gripper_pos, gripper_rot, args.save)


    gripper_pos, gripper_rot = ur5.get_end_effector_pose()
    gripper_pos = gripper_pos - constants.DESK_CENTER
    T_g_to_b = utils.pos_quat_to_transform(gripper_pos, gripper_rot)
    T_m_to_b = utils.pos_rot_to_transform(mug_param[1], utils.yaw_to_rot(mug_param[2]))
    T_m_to_g = np.matmul(np.linalg.inv(T_g_to_b), T_m_to_b)
    data["T"] = T_m_to_g

    ur5.gripper.close_gripper()

    input("Open gripper?")

    data["stop"] = True
    thread.join()

    save_place_contact_points(ur5, mug, tree, T_m_to_g, canon_mug, canon_tree, mug_param, tree_param, args.save)

    ur5.gripper.open_gripper()
    # ur5.plan_and_execute_joints_target(ur5.home_joint_values)


parser = argparse.ArgumentParser()
parser.add_argument("--show-perception", default=False, action="store_true")
parser.add_argument("--save", default=False, action="store_true")
main(parser.parse_args())
