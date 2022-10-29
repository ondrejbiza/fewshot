import argparse
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
import pybullet as pb

from pybullet_planning.pybullet_tools.ikfast.franka_panda.ik import PANDA_INFO, FRANKA_URDF
from pybullet_planning.pybullet_tools import utils as pu
from robot import Panda
import utils


def main(args):

    # setup sim and robot
    pu.connect(use_gui=True)
    pu.set_default_camera(distance=2)
    if args.real_time:
        pu.enable_real_time()
    else:
        pu.disable_real_time()
    pu.enable_gravity()
    pu.draw_global_system()

    with pu.LockRenderer():
        with pu.HideOutput():
            floor = pu.load_model("models/short_floor.urdf", fixed_base=True)
            robot_id = pu.load_model(FRANKA_URDF, fixed_base=True)
            pu.assign_link_colors(robot_id, max_colors=3, s=0.5, v=1.)

    robot = Panda(robot=robot_id)

    mug = pu.load_model("../data/mugs/test/0.urdf", fixed_base=False)
    tree = pu.load_model("../data/simple_trees/test/0.urdf", fixed_base=False)
    pu.set_pose(mug, pu.Pose(pu.Point(x=0.4, y=0.3, z=pu.stable_z(mug, floor)), pu.Euler(0., 0., np.pi / 4)))
    pu.set_pose(tree, pu.Pose(pu.Point(x=-0.4, y=0.3, z=pu.stable_z(tree, floor))))
    objects = [mug, tree]

    # get registered points on the warped canonical object
    pick_indices = np.load(args.pick_path)
    place_indices = np.load(args.place_path)
    print("Pick index:", pick_indices)
    print("Place indices:", place_indices)

    # get a point cloud
    pcs, _ = utils.observe_point_cloud(utils.RealSenseD415.CONFIG, [2, 3])

    # load canonical objects
    canon = {}
    with open("data/mugs_pca.pkl", "rb") as f:
        canon[2] = pickle.load(f)
    with open("data/simple_trees_pca.pkl", "rb") as f:
        canon[3] = pickle.load(f)

    # fit canonical objects to observed point clouds
    new_obj_1, _, _ = utils.planar_pose_warp_gd(canon[2]["pca"], canon[2]["canonical_obj"], pcs[2])
    new_obj_2, _, _ = utils.planar_pose_warp_gd(canon[3]["pca"], canon[3]["canonical_obj"], pcs[3], n_angles=1, object_size_reg=0.1)

    # get grasp
    position = new_obj_1[pick_indices]
    print("Position:", position)

    pose = pu.Pose(position, pu.Euler(roll=np.pi))

    pu.wait_if_gui()

    path = robot.plan_grasp(pose, 0.15, objects)
    robot.execute_path(path)
    robot.open_hand()

    pu.wait_if_gui()

    # path = robot.plan_grasp(pose, 0.1, [mug])
    path = robot.plan_grasp_naive(pose, 0.09)
    robot.execute_path(path)
    robot.close_hand()

    print("Picked object:", robot.get_picked_object(objects))

    pu.wait_if_gui()

    pose = robot.init_tool_pos, pu.get_link_pose(robot.robot, robot.tool_link)[1]
    path = robot.plan_motion(pose, [])
    robot.execute_path(path)

    print("Picked object:", robot.get_picked_object(objects))

    pu.wait_if_gui()

    points_1 = new_obj_1[place_indices[0]]
    points_2 = new_obj_2[place_indices[1]]

    # find best pair-wise fit
    # TODO: I want the target position and orientation here
    # Therefore, I need to get the estimated pose of the mug PC and multiply with the inverse.
    T, R, t = utils.best_fit_transform(points_1, points_2)
    print("Best fit spatial transform:")
    print(T)

    # create a spatial transformation matrix for the mug
    mug_pos, mug_rot = pu.get_pose(mug)
    mug_pos = np.array(mug_pos, dtype=np.float32)
    mug_rot = np.array(pb.getMatrixFromQuaternion(mug_rot), dtype=np.float32).reshape((3, 3))
    
    mug_T = np.concatenate([mug_rot, mug_pos[:, None]], axis=1)
    mug_T = np.concatenate([mug_T, np.array([[0., 0., 0., 1.]])], axis=0)
    print("Original mug spatial transform:")
    print(mug_T)

    # get a new mug pose
    new_mug_T = np.matmul(T, mug_T)
    print("New mug spatial transform:")
    print(new_mug_T)

    # make pose pybullet compatible
    new_mug_rot = new_mug_T[:3, :3]
    new_mug_pos = new_mug_T[:3, 3]
    new_mug_quat= Rotation.from_matrix(new_mug_rot).as_quat()

    pu.set_pose(mug, (new_mug_pos, new_mug_quat))

    pu.wait_if_gui()
    pu.disconnect()


parser = argparse.ArgumentParser()
parser.add_argument("--pick-path", default="data/clone_pick.npy")
parser.add_argument("--place-path", default="data/clone_place.npy")
parser.add_argument("-r", "--real-time", default=False, action="store_true")
main(parser.parse_args())
