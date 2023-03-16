import argparse
import copy as cp
import pickle
import threading
import time
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray
import rospy

from src import demo, utils, viz_utils
from src.real_world import perception
import src.real_world.utils as rw_utils
from src.real_world.ur5 import UR5
from src.real_world.point_cloud_proxy import PointCloudProxy
from src.real_world.simulation import Simulation


def worker(ur5: UR5, robotiq_id: int, source_id: int, data: Dict[str, Any]):

    trans_robotiq_to_tool0 = rw_utils.robotiq_to_tool0()

    while True:

        if data["stop"]:
            break

        trans_t0_to_b = utils.pos_quat_to_transform(*ur5.get_tool0_to_base())
        trans_t0_to_ws = np.linalg.inv(rw_utils.workspace_to_base()) @ trans_t0_to_b

        if data["trans_source_to_t0"] is not None:
            # Make the mug follow the gripper.
            trans_source_to_ws = np.matmul(trans_t0_to_ws, data["trans_source_to_t0"])
            utils.pb_set_pose(source_id, *utils.transform_to_pos_quat(trans_source_to_ws))

        trans_robotiq_to_ws = trans_t0_to_ws @ trans_robotiq_to_tool0
        robotiq_pos, robotiq_quat = utils.transform_to_pos_quat(trans_robotiq_to_ws)
        utils.pb_set_pose(robotiq_id, robotiq_pos, robotiq_quat)

        fract = ur5.gripper.get_open_fraction()
        utils.pb_set_joint_positions(robotiq_id, [0, 2, 4, 5, 6, 7], [fract, fract, fract, -fract, fract, -fract])

        time.sleep(0.1)


def save_pick_pose(observed_pc: NDArray[np.float32], canon_source: utils.CanonObj,
                   source_param: utils.ObjParam, ur5: UR5, trans_pre_t0_to_t0: NDArray[np.float32],
                   save_path: Optional[bool]=None):

    trans_t0_to_b = utils.pos_quat_to_transform(*ur5.get_tool0_to_base())
    trans_t0_tip_to_b = trans_t0_to_b @ rw_utils.tool0_tip_to_tool0()

    trans_source_to_ws = source_param.get_transform()
    trans_source_to_b = rw_utils.workspace_to_base() @ trans_source_to_ws
    trans_t0_tip_to_source = np.matmul(np.linalg.inv(trans_source_to_b), trans_t0_tip_to_b)

    # Grasp in a canonical frame.
    tip_pos, tip_quat = utils.transform_to_pos_quat(trans_t0_tip_to_source)

    # Warped object in a canonical frame.
    source_pc_canon = canon_source.to_pcd(source_param)

    # Index of the point on the canonical object closest to a point between the gripper fingers.
    dist = np.sqrt(np.sum(np.square(source_pc_canon - tip_pos), axis=1))
    index = np.argmin(dist)

    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump({
                "index": index,
                "pos": tip_pos,
                "quat": tip_quat,
                "trans_t0_to_b": trans_t0_to_b,
                "trans_pre_t0_to_t0": trans_pre_t0_to_t0,
                "observed_pc": observed_pc,
            }, f)


def save_pick_contact_points(observed_pc: NDArray[np.float32], robotiq_id: int, source_id: int,
                             canon_source: utils.CanonObj, source_param: utils.ObjParam,
                             ur5: UR5, trans_pre_t0_to_t0: NDArray[np.float32],
                             save_path: Optional[bool]=None):

    trans_t0_to_b = utils.pos_quat_to_transform(*ur5.get_tool0_to_base())

    # Get robotiq transform.
    pos, quat = utils.pb_get_pose(robotiq_id)
    trans_robotiq_to_ws = utils.pos_quat_to_transform(pos, quat)

    pos_robotiq_canon, index = demo.save_pick_contact_points(
        robotiq_id, source_id, trans_robotiq_to_ws, canon_source, source_param)

    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump({
                "index": index,
                "pos_robotiq": pos_robotiq_canon,
                "trans_t0_to_b": trans_t0_to_b,
                "trans_pre_t0_to_t0": trans_pre_t0_to_t0,
                "observed_pc": observed_pc,
            }, f)


def save_place_contact_points(
    target_pcd: NDArray, ur5: UR5, source_id: int, target_id: int, trans_source_to_t0: NDArray, trans_pre_t0_to_b: NDArray,
    canon_source: utils.CanonObj, canon_target: utils.CanonObj, source_param: utils.ObjParam, target_param: utils.ObjParam,
    delta: float=0.01, save_path: Optional[str]=None):

    trans_t0_to_b = utils.pos_quat_to_transform(*ur5.get_tool0_to_base())
    trans_source_to_b = trans_t0_to_b @ trans_source_to_t0

    trans_pre_source_to_b = trans_pre_t0_to_b @ trans_source_to_t0
    trans_pre_source_to_source = np.matmul(np.linalg.inv(trans_source_to_b), trans_pre_source_to_b)

    # The source object was perceived on the ground.
    # Move it to where the robot hand is now.
    source_param = cp.deepcopy(source_param)

    trans_source_to_b = np.matmul(trans_t0_to_b, trans_source_to_t0)
    trans_source_to_ws = np.linalg.inv(rw_utils.workspace_to_base()) @ trans_source_to_b
    src_pos, src_quat = utils.transform_to_pos_quat(trans_source_to_ws)
    source_param.position = src_pos
    source_param.quat = src_quat

    knns, deltas, i_2 = demo.save_place_nearby_points(
        source_id, target_id, canon_source, source_param, canon_target, target_param, delta, draw_spheres=True)

    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump({
                "knns": knns,
                "deltas": deltas,
                "target_indices": i_2,
                "trans_source_to_t0": trans_source_to_t0,
                "trans_t0_to_b": trans_t0_to_b,
                "trans_pre_t0_to_b": trans_pre_t0_to_b,
                "trans_pre_source_to_source": trans_pre_source_to_source,
                "observed_pc": target_pcd,
                "source_param": source_param,
                "target_param": target_param
            }, f)


def main(args):

    rospy.init_node("save_demo")
    pc_proxy = PointCloudProxy()

    ur5 = UR5(setup_planning=True)
    # ur5.plan_and_execute_joints_target(ur5.home_joint_values)
    ur5.gripper.open_gripper(position=70)

    platform_pcd = None
    if args.platform:
        cloud = pc_proxy.get_all()
        assert cloud is not None
        platform_pcd = perception.platform_segmentation(cloud)
        input("Platform captured. Continue? ")

    cloud = pc_proxy.get_all()
    assert cloud is not None

    sim = Simulation()

    if args.task == "mug_tree":
        canon_source = utils.CanonObj.from_pickle("data/230227_ndf_mugs_scale_large_pca_8_dim_alp_0_01.pkl")
        canon_target = utils.CanonObj.from_pickle("data/230228_simple_trees_scale_large_pca_8_dim_alp_0_01.pkl")
        canon_source.init_scale = 0.7
        canon_target.init_scale = 1.
        source_pcd, target_pcd = perception.mug_tree_segmentation(cloud, platform_pcd=platform_pcd)
    elif args.task == "bowl_on_mug":
        canon_source = utils.CanonObj.from_pickle("data/230227_ndf_bowls_scale_large_pca_8_dim_alp_0_01.pkl")
        canon_target = utils.CanonObj.from_pickle("data/230227_ndf_mugs_scale_large_pca_8_dim_alp_0_01.pkl")
        canon_source.init_scale = 0.8
        canon_target.init_scale = 0.7
        source_pcd, target_pcd = perception.bowl_mug_segmentation(cloud, platform_pcd=platform_pcd)
    elif args.task == "bottle_in_box":
        canon_source = utils.CanonObj.from_pickle("data/230227_ndf_bottles_scale_large_pca_8_dim_alp_0_01.pkl")
        canon_target = utils.CanonObj.from_pickle("data/230228_boxes_scale_large_pca_8_dim_alp_0_01.pkl")
        canon_source.init_scale = 1.
        canon_target.init_scale = 1.
        source_pcd, target_pcd = perception.bottle_box_segmentation(cloud, platform_pcd=platform_pcd)
    else:
        raise ValueError("Unknown task.")

    # Initial perception.
    out = perception.warping(
        source_pcd, target_pcd, canon_source, canon_target,
        tf_proxy=ur5.tf_proxy, moveit_scene=ur5.moveit_scene, source_save_decomposition=True,
        target_save_decomposition=True, add_source_to_planning_scene=True, add_target_to_planning_scene=True,
        rviz_pub=ur5.rviz_pub
    )
    source_pcd_complete, source_param, target_pcd_complete, target_param = out

    # Setup simulation with what we see in the real world.
    source_id = sim.add_object("tmp_source.urdf", source_param.position, source_param.quat)
    target_id = sim.add_object("tmp_target.urdf", target_param.position, target_param.quat, fixed_base=True)
    robotiq_id = sim.add_object("data/robotiq.urdf", np.array([0., 0., 0.]), np.array([0., 0., 0., 1.]))

    # Continuously update the gripper position in simulation.
    data = {
        "trans_source_to_t0": None,
        "stop": False,
    }
    thread = threading.Thread(target=worker, args=(ur5, robotiq_id, source_id, data))
    thread.start()

    input("Save place waypoint?")
    trans_pre_t0_to_b = utils.pos_quat_to_transform(*ur5.get_tool0_to_base())

    input("Close gripper?")
    ur5.gripper.close_gripper()

    # Compute relation between the gripper and the object.
    trans_t0_to_b = utils.pos_quat_to_transform(*ur5.get_tool0_to_base())
    trans_source_to_ws = source_param.get_transform()
    trans_source_to_b = rw_utils.workspace_to_base() @ trans_source_to_ws
    trans_source_to_t0 = np.linalg.inv(trans_t0_to_b) @ trans_source_to_b
    data["trans_source_to_t0"] = trans_source_to_t0

    trans_pre_t0_to_t0 = np.matmul(np.linalg.inv(trans_t0_to_b), trans_pre_t0_to_b)

    input("Take in-hand image?")

    # Refine gripper-object transform using a second point cloud.
    # The object might have moved as the gripper fingers closed.
    cloud = pc_proxy.get_all()
    assert cloud is not None

    source_param, source_pcd_complete, trans_source_to_t0, in_hand_pcd = perception.reestimate_tool0_to_source(
        cloud, ur5, robotiq_id, sim, canon_source, source_param, trans_source_to_t0)
    data["trans_source_to_t0"] = trans_source_to_t0

    viz_utils.show_pcds_plotly({
        "pcd": in_hand_pcd,
        "completed": source_pcd_complete
    })

    if args.pick_contacts:
        save_pick_contact_points(
            source_pcd, robotiq_id, source_id, canon_source, source_param, ur5, trans_pre_t0_to_t0, args.pick_save_path)
    else:
        save_pick_pose(source_pcd, canon_source, source_param, ur5, trans_pre_t0_to_t0, args.pick_save_path)

    input("Save place waypoint?")
    trans_pre_t0_to_b = utils.pos_quat_to_transform(*ur5.get_tool0_to_base())

    input("Save place pose?")
    save_place_contact_points(
        target_pcd, ur5, source_id, target_id, trans_source_to_t0, trans_pre_t0_to_b, canon_source, canon_target,
        source_param, target_param, save_path=args.place_save_path)

    # Reset.
    # input("Open gripper?")
    ur5.gripper.open_gripper()
    # ur5.plan_and_execute_joints_target(ur5.home_joint_values)

    data["stop"] = True
    thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Collect a demonstration.")
    parser.add_argument("task", type=str, help="[mug_tree, bowl_on_mug, bottle_in_box]")
    parser.add_argument("pick_save_path", type=str)
    parser.add_argument("place_save_path", type=str)
    parser.add_argument("-c", "--pick-contacts", default=False, action="store_true")
    parser.add_argument("-p", "--platform", default=False, action="store_true",
                        help="First take a point cloud of a platform. Then subtract the platform from the next point cloud.")
    main(parser.parse_args())
