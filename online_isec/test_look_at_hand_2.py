import numpy as np
import pybullet as pb
import rospy
from scipy.spatial.transform import Rotation
import time
import pickle

from online_isec.ur5 import UR5
import online_isec.utils as isec_utils
from pybullet_planning.pybullet_tools import utils as pu
from online_isec import constants
import utils
from online_isec.point_cloud_proxy_sync import RealsenseStructurePointCloudProxy
import viz_utils
from online_isec import perception
from online_isec.moveit_plan_pick_place_plan import pick
from online_isec.simulation import Simulation


def main():

    rospy.init_node("test_robotiq_in_pybullet")
    pc_proxy = RealsenseStructurePointCloudProxy()

    ur5 = UR5(setup_planning=True)
    ur5.plan_and_execute_joints_target(ur5.home_joint_values)
    ur5.gripper.open_gripper(position=100)

    mug_pc_complete, mug_param, tree_pc_complete, tree_param, canon_mug, canon_tree = perception.mug_tree_perception(
        pc_proxy, np.array(constants.DESK_CENTER), ur5.tf_proxy, ur5.moveit_scene,
        add_mug_to_planning_scene=False, add_tree_to_planning_scene=False, rviz_pub=ur5.rviz_pub
    )

    pick(mug_pc_complete, mug_param, ur5, "data/220129_real_pick_clone.pkl")

    sim = Simulation()

    mug_pc_1 = perception.perceive_mug_in_hand(pc_proxy, sim, ur5)
    # viz_utils.o3d_visualize(utils.create_o3d_pointcloud(mug_pc_1))

    orig_joints = np.copy(ur5.joint_values)
    new_joints = np.copy(orig_joints)
    new_joints[-1] += (2 * np.pi) / 3
    ur5.plan_and_execute_joints_target(new_joints)

    mug_pc_2 = perception.perceive_mug_in_hand(pc_proxy, sim, ur5)
    # viz_utils.o3d_visualize(utils.create_o3d_pointcloud(mug_pc_2))
    ur5.plan_and_execute_joints_target(orig_joints)

    new_joints = np.copy(orig_joints)
    new_joints[-1] -= (2 * np.pi) / 3
    ur5.plan_and_execute_joints_target(new_joints)

    mug_pc_3 = perception.perceive_mug_in_hand(pc_proxy, sim, ur5)
    # viz_utils.o3d_visualize(utils.create_o3d_pointcloud(mug_pc_3))
    ur5.plan_and_execute_joints_target(orig_joints)

    # One would think we need -(2pi)/3 here and one would be wrong.
    rot2 = Rotation.from_euler("z", (2 * np.pi) / 3).as_matrix()
    mug_pc_2 = np.matmul(mug_pc_2, rot2.T)

    rot3 = Rotation.from_euler("z", - (2 * np.pi) / 3).as_matrix()
    mug_pc_3 = np.matmul(mug_pc_3, rot3.T)

    viz_utils.show_scene({
        1: mug_pc_1,
        2: mug_pc_2,
        3: mug_pc_3,
    })

    mug_pc = np.concatenate([mug_pc_1, mug_pc_2, mug_pc_3])
    mug_pc, _ = utils.farthest_point_sample(mug_pc, 2000)

    mug_pc_complete, _, mug_param = utils.planar_pose_warp_gd(canon_mug["pca"], canon_mug["canonical_obj"], mug_pc, object_size_reg=0.05, n_angles=12)
    viz_utils.show_scene({
        1: mug_pc_complete,
    }, background=mug_pc)

    ur5.gripper.open_gripper()
    ur5.plan_and_execute_joints_target(ur5.home_joint_values)


if __name__ == "__main__":
    main()
