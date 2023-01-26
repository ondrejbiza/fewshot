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

        pos, quat = ur5.get_end_effector_pose()
        pos = pos - constants.DESK_CENTER

        if data["T"] is not None:
            T_g = utils.pos_quat_to_transform(pos, quat)
            T = np.matmul(T_g, data["T"])
            m_pos, m_quat = utils.transform_to_pos_quat(T)
            pu.set_pose(mug, (m_pos, m_quat))

        pu.set_position(sphere, pos[0], pos[1], pos[2])
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


def save_place_contact_points(ur5, mug, tree, T_g_to_m, canon_mug, mug_param, canon_tree, save: bool):

    spheres = []

    pb.performCollisionDetection()

    cols = pb.getClosestPoints(mug, tree, 0.01)

    for s in spheres:
        pu.remove_body(s)
    spheres = []

    for col in cols:
        pos = col[6]
        with pu.HideOutput():
            s = pu.load_model("../data/sphere.urdf")
            pu.set_pose(s, pu.Pose(pu.Point(*pos)))
        spheres.append(s)

    pos_1 = [col[5] for col in cols]
    pos_2 = [col[6] for col in cols]

    pos_1 = np.stack(pos_1, axis=0).astype(np.float32)
    pos_2 = np.stack(pos_2, axis=0).astype(np.float32)

    pos, quat = ur5.get_end_effector_pose()
    T_g = utils.pos_quat_to_transform(pos - constants.DESK_CENTER, quat)
    T = np.matmul(T_g, T_g_to_m)

    tmp = utils.canon_to_pc(canon_mug, mug_param)
    tmp_2 = utils.transform_pointcloud_2(pos_2, np.linalg.inv(T))

    knns, deltas = get_knn_and_deltas(tmp, tmp_2)

    dist_2 = np.sqrt(np.sum(np.square(canon_tree["canonical_obj"][:, None] - tmp_2[None]), axis=2))
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
    ur5 = UR5(setup_planning=True)
    ur5.plan_and_execute_joints_target(ur5.home_joint_values)

    pc_proxy = RealsenseStructurePointCloudProxy()
    mug_pc_complete, mug_param, tree_pc_complete, tree_param = perception.mug_tree_perception(
        pc_proxy, np.array(constants.DESK_CENTER)
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
        np.linalg.inv(utils.yaw_to_rot(mug_param[2])),
        Rotation.from_quat(gripper_rot).as_matrix()
    )
    print("%", tmp)
    gripper_rot = Rotation.from_matrix(tmp).as_quat()
    save_pick_pose(mug_pc_complete, gripper_pos, gripper_rot, args.save)

    T_g = utils.pos_quat_to_transform(gripper_pos, gripper_rot)

    mug_pos, mug_rot = pu.get_pose(mug)
    T_m = utils.pos_quat_to_transform(mug_pos, mug_rot)

    T = utils.compute_relative_transform(T_g, T_m)
    data["T"] = T

    ur5.gripper.close_gripper()

    input("Open gripper?")

    data["stop"] = True
    thread.join()

    save_place_contact_points(ur5, mug, tree, T, canon_mug, mug_param, canon_tree, args.save)

    ur5.gripper.open_gripper()
    ur5.plan_and_execute_joints_target(ur5.home_joint_values)


parser = argparse.ArgumentParser()
parser.add_argument("--show-perception", default=False, action="store_true")
parser.add_argument("--save", default=False, action="store_true")
main(parser.parse_args())
