import argparse
import os
import pickle

import numpy as np
from numpy.typing import NDArray
import rospy

from src import utils, viz_utils
from src.real_world import constants, perception
import src.real_world.utils as rw_utils
from src.real_world.point_cloud_proxy import PointCloudProxy
from src.real_world.ur5 import UR5
from src.real_world.simulation import Simulation

EASY_MOTION_TRIES = 3
HARD_MOTION_TRIES = 5


def pick_simple(source_pcd_complete: NDArray, source_param: utils.ObjParam, 
                ur5: UR5, post_pick_delta: float, load_path: str) -> NDArray:
    """Pick object by following a single vertex position and a pre-recorded pose."""
    with open(load_path, "rb") as f:
        data = pickle.load(f)
    index, t0_tip_quat = data["index"], data["quat"]
    t0_tip_pos_in_ws = source_pcd_complete[index]
    trans_pre_t0_to_t0 = data["trans_pre_t0_to_t0"]

    t0_tip_pos_in_base = t0_tip_pos_in_ws + constants.DESK_CENTER
    t0_tip_rot = np.matmul(
        utils.quat_to_rotm(source_param.quat),
        utils.quat_to_rotm(t0_tip_quat)
    )
    t0_tip_quat = utils.rotm_to_quat(t0_tip_rot)

    trans_t0_tip_to_b = utils.pos_quat_to_transform(t0_tip_pos_in_base, t0_tip_quat)
    trans_t0_to_b = trans_t0_tip_to_b @ np.linalg.inv(rw_utils.tool0_tip_to_tool0())

    trans_post_t0_to_b = np.copy(trans_t0_to_b)
    trans_post_t0_to_b[2, 3] += post_pick_delta
    trans_pre_t0_to_b = np.matmul(trans_t0_to_b, trans_pre_t0_to_t0)
    # trans_pre_t0_to_b = utils.pos_quat_to_transform(*rw_utils.move_hand_back(*utils.transform_to_pos_quat(trans_t0_to_b), 0.1))

    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(trans_pre_t0_to_b), num_plans=HARD_MOTION_TRIES)

    # Remove source object from planning scene.
    ur5.moveit_scene.remove_object("source")

    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(trans_t0_to_b), num_plans=EASY_MOTION_TRIES)
    ur5.gripper.close_gripper()

    # Add source object back to the planning scene.
    pos, quat = utils.transform_to_pos_quat(
        rw_utils.desk_obj_param_to_base_link_T(source_param.position, source_param.quat, np.array(constants.DESK_CENTER), ur5.tf_proxy))
    ur5.moveit_scene.add_object("tmp_source.stl", "source", pos, quat)

    # Lock source object to flange in the moveit scene.
    ur5.moveit_scene.attach_object("source")

    # Calculate source object to gripper transform at the point when we grasp it.
    trans_t0_to_b = utils.pos_quat_to_transform(*ur5.get_tool0_to_base())
    trans_source_to_ws = source_param.get_transform()
    trans_source_to_b = rw_utils.workspace_to_base() @ trans_source_to_ws
    trans_source_to_t0 = np.linalg.inv(trans_t0_to_b) @ trans_source_to_b

    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(trans_post_t0_to_b), num_plans=EASY_MOTION_TRIES)

    return trans_source_to_t0


def pick_contacts(ur5: UR5, canon_source: utils.CanonObj, source_param: utils.ObjParam, post_pick_delta: float, load_path: str):

    with open(load_path, "rb") as f:
        d = pickle.load(f)
    index = d["index"]
    pos_robotiq = d["pos_robotiq"]
    trans_pre_t0_to_t0 = d["trans_pre_t0_to_t0"]

    source_pcd_complete = canon_source.to_pcd(source_param)
    source_points = source_pcd_complete[index]

    # Pick pose in canonical frame..
    trans, _, _ = utils.best_fit_transform(pos_robotiq, source_points)
    # Pick pose in workspace frame.
    trans_robotiq_to_ws = source_param.get_transform() @ trans

    trans_t0_to_ws = trans_robotiq_to_ws @ np.linalg.inv(rw_utils.robotiq_to_tool0())
    trans_t0_to_b = rw_utils.workspace_to_base() @ trans_t0_to_ws

    # Either lift the hand 25 cm above the workspace,
    # or two cm from the current position if its higher than 25 cm.
    trans_t0_to_ws = np.linalg.inv(rw_utils.workspace_to_base()) @ trans_t0_to_b
    trans_post_t0_to_ws = rw_utils.get_post_pick_pose(trans_t0_to_ws)
    trans_post_t0_to_b = rw_utils.workspace_to_base() @ trans_post_t0_to_ws

    trans_pre_t0_to_b = np.matmul(trans_t0_to_b, trans_pre_t0_to_t0)

    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(trans_pre_t0_to_b), num_plans=HARD_MOTION_TRIES)

    # Remove mug from planning scene.
    ur5.moveit_scene.remove_object("source")

    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(trans_t0_to_b), num_plans=EASY_MOTION_TRIES)
    ur5.gripper.close_gripper()

    # Calcualte mug to gripper transform at the point when we grasp it.
    trans_t0_to_b = utils.pos_quat_to_transform(*ur5.get_tool0_to_base())
    trans_source_to_ws = source_param.get_transform()
    trans_source_to_b = rw_utils.workspace_to_base() @ trans_source_to_ws
    trans_source_to_t0 = np.linalg.inv(trans_t0_to_b) @ trans_source_to_b

    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(trans_post_t0_to_b), num_plans=HARD_MOTION_TRIES)

    return trans_source_to_t0


def place(
    ur5: UR5, canon_source: utils.CanonObj, canon_target: utils.CanonObj,
    source_param: utils.ObjParam, target_param: utils.ObjParam,
    trans_source_to_t0: NDArray, sim: Simulation, load_path: str):

    with open(load_path, "rb") as f:
        place_data = pickle.load(f)

    knns = place_data["knns"]
    deltas = place_data["deltas"]
    target_indices = place_data["target_indices"]
    trans_pre_source_to_source = place_data["trans_pre_source_to_source"]

    source_orig = canon_source.to_pcd(source_param)
    target_orig = canon_target.to_pcd(target_param)

    anchors = source_orig[knns]
    contacts_source = np.mean(anchors + deltas, axis=1)
    contacts_target = target_orig[target_indices]

    trans_new_source_to_target, _, _ = utils.best_fit_transform(contacts_source, contacts_target)
    trans_target_to_b = utils.pos_quat_to_transform(target_param.position + constants.DESK_CENTER, target_param.quat)
    trans_new_source_to_b = np.matmul(trans_target_to_b, trans_new_source_to_target)

    # Wiggle mug out of potential collision.
    # sim.remove_all_objects()
    # source_id = sim.add_object("tmp_source.urdf", *utils.transform_to_pos_quat(trans_new_source_to_b))
    # target_id = sim.add_object("tmp_target.urdf", *utils.transform_to_pos_quat(trans_target_to_b), fixed_base=True)
    # new_m_pos, new_m_quat = sim.wiggle(mug_id, tree_id)
    # T_new_m_to_b = utils.pos_quat_to_transform(new_m_pos, new_m_quat)

    trans_pre_new_source_to_b = trans_new_source_to_b @ trans_pre_source_to_source

    trans_t0_to_b = np.matmul(trans_new_source_to_b, np.linalg.inv(trans_source_to_t0))
    trans_pre_t0_to_b = np.matmul(trans_pre_new_source_to_b, np.linalg.inv(trans_source_to_t0))

    # I believe this is the wrong approach to do the pre-place pose:
    # trans_pre_t0_to_b = np.matmul(trans_t0_to_b, trans_pre_t0_to_t0)

    # Remove mug and tree from the planning scene.
    # ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(trans_pre_t0_to_b), num_plans=HARD_MOTION_TRIES)

    ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(trans_t0_to_b), num_plans=EASY_MOTION_TRIES)

    # ur5.plan_and_execute_pose_target(*utils.transform_to_pos_quat(trans_pre_t0_to_b), num_plans=EASY_MOTION_TRIES)


def main(args):

    rospy.init_node("pick_place")
    pc_proxy = PointCloudProxy()

    ur5 = UR5(setup_planning=True)
    ur5.plan_and_execute_joints_target(ur5.home_joint_values)
    ur5.gripper.open_gripper(position=20)

    platform_pcd = None
    if args.platform:
        cloud = pc_proxy.get_all()
        assert cloud is not None
        platform_pcd = perception.platform_segmentation(cloud)
        input("Platform captured. Continue? ")

    cloud = pc_proxy.get_all()

    sim = Simulation()

    if args.task == "mug_tree":
        canon_source = utils.CanonObj.from_pickle(constants.NDF_MUGS_PCA_PATH)
        canon_target = utils.CanonObj.from_pickle(constants.SIMPLE_TREES_PCA_PATH)
        canon_source.init_scale = constants.NDF_MUGS_INIT_SCALE
        canon_target.init_scale = constants.SIMPLE_TREES_INIT_SCALE
        source_pcd, target_pcd = perception.mug_tree_segmentation(cloud, platform_pcd=platform_pcd)
        
    elif args.task == "bowl_on_mug":
        canon_source = utils.CanonObj.from_pickle(constants.NDF_BOWLS_PCA_PATH)
        canon_target = utils.CanonObj.from_pickle(constants.NDF_MUGS_PCA_PATH)
        canon_source.init_scale = constants.NDF_BOWLS_INIT_SCALE
        canon_target.init_scale = constants.NDF_MUGS_INIT_SCALE
        source_pcd, target_pcd = perception.bowl_mug_segmentation(cloud, platform_pcd=platform_pcd)

    elif args.task == "bottle_in_box":
        canon_source = utils.CanonObj.from_pickle(constants.NDF_BOTTLES_PCA_PATH)
        canon_target = utils.CanonObj.from_pickle(constants.BOXES_PCA_PATH)
        canon_source.init_scale = constants.NDF_BOTTLES_INIT_SCALE
        canon_target.init_scale = constants.BOXES_INIT_SCALE
        source_pcd, target_pcd = perception.bottle_box_segmentation(cloud, platform_pcd=platform_pcd)

    elif args.task == "brush_in_bowl":
        canon_source = utils.CanonObj.from_pickle(constants.NDF_BRUSH_PCA_PATH_3)
        canon_target = utils.CanonObj.from_pickle(constants.NDF_BOWLS_PCA_PATH)
        canon_source.init_scale = constants.NDF_BRUSHES_INIT_SCALE
        canon_target.init_scale = constants.NDF_BOWLS_INIT_SCALE      
        source_pcd, target_pcd = perception.bottle_box_segmentation(cloud, platform_pcd=platform_pcd)

    elif args.task == "brush_on_cube":
        canon_source = utils.CanonObj.from_pickle(constants.NDF_BRUSH_PCA_PATH_3)
        canon_target = utils.CanonObj.from_pickle(constants.NDF_CUBE_PCA_PATH)
        canon_source.init_scale = constants.NDF_BRUSHES_INIT_SCALE
        canon_target.init_scale = constants.NDF_CUBE_INIT_SCALE
        source_pcd, target_pcd = perception.bottle_box_segmentation(cloud, platform_pcd=platform_pcd)

    elif args.task == "cubiods":
        canon_source = utils.CanonObj.from_pickle()

    else:
        raise ValueError("Unknown task.")

    out = perception.warping(
        source_pcd, target_pcd, canon_source, canon_target,
        source_any_rotation=args.any_rotation,
        target_any_rotation=args.target_any_rotation,
        tf_proxy=ur5.tf_proxy, moveit_scene=ur5.moveit_scene, source_save_decomposition=False,
        target_save_decomposition=False, add_source_to_planning_scene=True, add_target_to_planning_scene=True,
        rviz_pub=ur5.rviz_pub, grow_source_object=True
    )
    source_pcd_complete, source_param, target_pcd_complete, target_param = out

    post_pick_delta = 0.1
    if args.platform:
        post_pick_delta = 0.1

    if args.pick_contacts:
        trans_source_to_t0 = pick_contacts(ur5, canon_source, source_param, post_pick_delta, args.pick_load_path)
    else:
        trans_source_to_t0 = pick_simple(source_pcd_complete, source_param, ur5, post_pick_delta, args.pick_load_path)

    # Take an in-hand image.
    if args.platform:
        input("Platform removed?")

    cloud = pc_proxy.get_all()

    robotiq_id = sim.add_object("data/robotiq.urdf", np.array([0., 0., 0.]), np.array([0., 0., 0., 1.]))
    source_param, source_pcd_complete, trans_source_to_t0, in_hand_pcd = perception.reestimate_tool0_to_source(
        cloud, ur5, robotiq_id, sim, canon_source, source_param, trans_source_to_t0)
    sim.remove_object(robotiq_id)

    viz_utils.show_pcds_plotly({
        "pcd": in_hand_pcd,
        "completed": source_pcd_complete
    })

    # # Add mug back to the planning scene.
    pos, quat = utils.transform_to_pos_quat(
        rw_utils.desk_obj_param_to_base_link_T(source_param.position, source_param.quat, np.array(constants.DESK_CENTER), ur5.tf_proxy))
    ur5.moveit_scene.add_object("tmp_source.stl", "source", pos, quat)

    # # Lock mug to flange in the moveit scene.
    ur5.moveit_scene.attach_object("source")

    ur5.plan_and_execute_joints_target(ur5.home_joint_values)

    # Disable all collision checking for objects.
    ur5.moveit_scene.detach_object("source")
    ur5.moveit_scene.remove_object("source")
    ur5.moveit_scene.remove_object("target")

    i = 0
    while True:
        tmp_place_path = args.place_load_path.split(".")
        tmp_place_path[-2] += "_{:d}".format(i)
        tmp_place_path = ".".join(tmp_place_path)

        if not os.path.isfile(tmp_place_path):
            break

        place(ur5, canon_source, canon_target, source_param, target_param, trans_source_to_t0, sim, tmp_place_path)
        i += 1

    ur5.gripper.open_gripper()
    ur5.plan_and_execute_joints_target(ur5.home_joint_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help=constants.TASKS_DESCRIPTION)
    parser.add_argument("pick_load_path", type=str)
    parser.add_argument("place_load_path", type=str)

    parser.add_argument("-a", "--any-rotation", default=False, action="store_true",
                        help="Try to determine SE(3) object pose. Otherwise, determine a planar pose.")
    parser.add_argument("-t", "--target-any-rotation", default=False, action="store_true")
    parser.add_argument("-c", "--pick-contacts", default=False, action="store_true")
    parser.add_argument("-p", "--platform", default=False, action="store_true",
                        help="First take a point cloud of a platform. Then subtract the platform from the next point cloud.")

    parser.add_argument("--ablate-no-mug-warping", default=False, action="store_true")

    main(parser.parse_args())
