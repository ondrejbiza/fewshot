import time
import os
import numpy as np
import pybullet as pb
from pybullet_planning.pybullet_tools import kuka_primitives
from pybullet_planning.pybullet_tools import utils as pu


def replace_orientation(pose, roll, pitch, yaw):

    return pu.Pose(pose[0], pu.Euler(roll, pitch, yaw))


def main():

    pu.connect(use_gui=True)
    pu.set_default_camera(distance=2)
    pu.disable_real_time()
    pu.draw_global_system()
    with pu.HideOutput():
        robot = pu.load_model(pu.DRAKE_IIWA_URDF) # KUKA_IIWA_URDF | DRAKE_IIWA_URDF
        floor = pu.load_model('models/short_floor.urdf')

    mug = pu.load_model("../data/mugs/0.urdf", fixed_base=False)
    pu.set_pose(mug, pu.Pose(pu.Point(x=0.4, y=0.3, z=pu.stable_z(mug, floor)), pu.Euler(0., 0., 0.)))

    tree = pu.load_model("../data/mug_tree/mug_tree.urdf")
    pu.set_pose(tree, pu.Pose(pu.Point(x=-0.4, y=0.3, z=pu.stable_z(tree, floor))))

    link_pose = pu.get_link_pose(mug, 0)
    link_pose = replace_orientation(link_pose, np.pi, 0., 0.)
    p = pu.Pose(point=[0, 0, 0.01])
    link_pose = pu.multiply(p, link_pose)

    conf0 = kuka_primitives.BodyConf(robot)

    pu.dump_world()
    saved_world = pu.WorldSaver()

    free_motion_fn = kuka_primitives.get_free_motion_gen(
        robot, fixed=([floor, tree, mug]), teleport=False)
    holding_motion_fn = kuka_primitives.get_holding_motion_gen(
        robot, fixed=[floor, tree], teleport=False)

    q_grasp = pu.inverse_kinematics(robot, kuka_primitives.get_tool_link(robot), link_pose)
    conf1 = kuka_primitives.BodyConf(robot, q_grasp)
    path1, = free_motion_fn(conf0, conf1)

    # path2, = holding_motion_fn(conf1, conf0, mug, link_pose)
    # command = kuka_primitives.Command(path1.body_paths + path2.body_paths)
    command = kuka_primitives.Command(path1.body_paths)

    saved_world.restore()
    pu.update_state()

    command.refine(num_steps=10).execute(time_step=0.005)

    pu.wait_if_gui()
    pu.disconnect()


main()
