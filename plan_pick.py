import argparse
import pickle
import numpy as np

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
    pu.set_pose(mug, pu.Pose(pu.Point(x=0.4, y=0.3, z=pu.stable_z(mug, floor)), pu.Euler(0., 0., np.pi / 4)))

    # get registered points on the warped canonical object
    indices = np.load(args.load_path)
    print("Registered index:", indices)

    # get a point cloud
    pcs, _ = utils.observe_point_cloud(utils.RealSenseD415.CONFIG, [2])

    # load canonical objects
    canon = {}
    with open("data/mugs_pca.pkl", "rb") as f:
        canon[2] = pickle.load(f)

    # fit canonical objects to observed point clouds
    new_obj_1, _, _ = utils.planar_pose_warp_gd(canon[2]["pca"], canon[2]["canonical_obj"], pcs[2])

    # get grasp
    position = new_obj_1[indices]
    print("Position:", position)

    pose = pu.Pose(position, pu.Euler(roll=np.pi))

    pu.wait_if_gui()

    path = robot.plan_grasp(pose, 0.15, [mug])
    robot.execute_path(path)
    robot.open_hand()

    pu.wait_if_gui()

    # path = robot.plan_grasp(pose, 0.1, [mug])
    path = robot.plan_grasp_naive(pose, 0.09)
    robot.execute_path(path)
    robot.close_hand()

    print("Picked object:", robot.get_picked_object([mug]))

    pu.wait_if_gui()

    pose = robot.init_tool_pos, pu.get_link_pose(robot.robot, robot.tool_link)[1]
    path = robot.plan_motion(pose, [])
    robot.execute_path(path)

    print("Picked object:", robot.get_picked_object([mug]))

    pu.wait_if_gui()

    pu.disconnect()


parser = argparse.ArgumentParser()
parser.add_argument("--load-path", default="data/clone_pick.npy")
parser.add_argument("-r", "--real-time", default=False, action="store_true")
main(parser.parse_args())
