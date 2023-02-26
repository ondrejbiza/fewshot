import argparse
import pickle
import threading
import time
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray
import rospy

from src import demo, utils
from src.real_world import constants, perception
import src.real_world.utils as rw_utils
from src.real_world.ur5 import UR5
from src.real_world.point_cloud_proxy_sync import PointCloudProxy
from src.real_world.simulation import Simulation


def worker(ur5: UR5, sphere: int, source_id: int, data: Dict[str, Any]):

    while True:

        if data["stop"]:
            break

        gripper_pos, gripper_quat = ur5.get_end_effector_pose()
        gripper_pos = gripper_pos - constants.DESK_CENTER

        if data["T"] is not None:
            # Make the mug follow the gripper.
            T_g_to_b = utils.pos_quat_to_transform(gripper_pos, gripper_quat)
            T_src_to_b = np.matmul(T_g_to_b, data["T"])
            src_pos, src_quat = utils.transform_to_pos_quat(T_src_to_b)
            utils.pb_set_pose(source_id, src_pos, src_quat)

        # Mark the gripper with a sphere.
        utils.pb_set_pose(sphere, gripper_pos, np.array([0., 0., 0., 1.]))
        time.sleep(0.1)


def save_pick_pose(observed_pc: NDArray[np.float32], canon_mug: utils.CanonObj,
                   mug_param: utils.ObjParam, ur5: UR5, save_path: Optional[bool]=None):

    T_g_to_b = utils.pos_quat_to_transform(*ur5.get_end_effector_pose())
    T_m_to_b = utils.pos_quat_to_transform(mug_param.position + constants.DESK_CENTER, mug_param.quat)
    T_g_to_m = np.matmul(np.linalg.inv(T_m_to_b), T_g_to_b)

    # Grasp in a canonical frame.
    g_to_m_pos, g_to_m_quat = utils.transform_to_pos_quat(T_g_to_m)

    # Warped object in a canonical frame.
    mug_pc_complete = canon_mug.to_pcd(mug_param)

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
    target_pcd: NDArray, ur5: UR5, source_id: int, target_id: int, T_src_to_g: NDArray, T_g_pre_to_b: NDArray,
    canon_source: utils.CanonObj, canon_target: utils.CanonObj, source_param: utils.ObjParam, target_param: utils.ObjParam,
    delta: float=0.1, save_path: Optional[str]=None):

    T_g_to_b = utils.pos_quat_to_transform(*ur5.get_end_effector_pose())
    T_g_pre_to_g = np.matmul(np.linalg.inv(T_g_to_b), T_g_pre_to_b)

    knns, deltas, i_2 = demo.save_place_nearby_points(
        source_id, target_id, canon_source, source_param, canon_target, target_param, delta, draw_spheres=True)

    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump({
                "knns": knns,
                "deltas": deltas,
                "target_indices": i_2,
                "T_g_to_b": T_g_to_b,
                "T_g_pre_to_g": T_g_pre_to_g,
                "observed_pc": target_pcd,
            }, f)


def main(args):

    rospy.init_node("save_demo")
    pc_proxy = PointCloudProxy()

    ur5 = UR5(setup_planning=True)
    # ur5.plan_and_execute_joints_target(ur5.home_joint_values)
    ur5.gripper.open_gripper(position=70)

    cloud = pc_proxy.get_all()
    assert cloud is not None

    sim = Simulation()

    if args.task == "mug_tree":
        out = perception.mug_tree_perception(
            cloud, ur5.tf_proxy, ur5.moveit_scene,
            add_mug_to_planning_scene=True, add_tree_to_planning_scene=True, rviz_pub=ur5.rviz_pub,
            mug_save_decomposition=True, tree_save_decomposition=True
        )
    else:
        raise NotImplementedError()

    source_pcd_complete, source_param, target_pcd_complete, target_param, canon_source, canon_target, source_pcd, target_pcd = out

    # Setup simulation with what we see in the real world.
    source_id = sim.add_object("tmp_source.urdf", source_param.position, source_param.quat)
    target_id = sim.add_object("tmp_target.urdf", target_param.position, target_param.quat, fixed_base=True)
    sphere_id = sim.add_object("data/sphere.urdf", np.array([0., 0., 0.]), np.array([0., 0., 0., 1.]))

    # Continuously update the gripper position in simulation.
    data = {
        "T": None,
        "stop": False,
    }
    thread = threading.Thread(target=worker, args=(ur5, sphere_id, source_id, data))
    thread.start()

    input("Close gripper?")
    ur5.gripper.close_gripper()
    save_pick_pose(source_pcd, canon_source, source_param, ur5, args.pick_save_path + ".pkl")

    # Calculate mug to gripper transform. Transmit it to simulation.
    T_g_to_b = utils.pos_quat_to_transform(*ur5.get_end_effector_pose())
    T_ws_to_b = rw_utils.workspace_to_base()
    T_g_to_ws = np.matmul(np.linalg.inv(T_ws_to_b), T_g_to_b)

    T_src_to_ws = utils.pos_quat_to_transform(source_param.position, source_param.quat)
    T_src_to_g = np.matmul(np.linalg.inv(T_g_to_ws), T_src_to_ws)
    data["T"] = T_src_to_g

    # Save any number of waypoints.
    input("Save place waypoint?")
    T_g_pre_to_b = utils.pos_quat_to_transform(*ur5.get_end_effector_pose())

    input("Save place pose?")
    save_place_contact_points(
        target_pcd, ur5, source_id, target_id, T_src_to_g, T_g_pre_to_b, canon_source, canon_target,
        source_param, target_param, save_path=args.place_save_path + ".pkl")

    # Reset.
    # input("Open gripper?")
    ur5.gripper.open_gripper()
    # ur5.plan_and_execute_joints_target(ur5.home_joint_values)

    data["stop"] = True
    thread.join()


parser = argparse.ArgumentParser("Collect a demonstration.")
parser.add_argument("task", type=str, help="[mug_tree, bowl_on_mug, bottle_in_box]")
parser.add_argument("pick_save_path", type=str, help="Postfix added automatically.")
parser.add_argument("place_save_path", type=str, help="Postfix added automatically.")
main(parser.parse_args())
