import time
import os
import numpy as np
import pybullet as pb
from pybullet_planning.pybullet_tools import kuka_primitives
from pybullet_planning.pybullet_tools import utils as pu


def get_mug_grasp(mug, robot):

    # TODO: take into account the orientation of the mug
    base_position = pu.Point(*pb.getBasePositionAndOrientation(mug)[0])
    ap_position = pu.Point(*pb.getJointInfo(mug, 0)[14])
    pick_action = ap_position
    pick_action[2] += 0.02

    grasp = kuka_primitives.BodyGrasp(
        mug, 
        pu.Pose(pick_action, pu.Euler(0., np.pi, 0.)),  # TODO: I don't understand how the grasp pose is specified.
        pu.Pose(0.1 * pu.Point(z=1)),
        robot,
        kuka_primitives.get_tool_link(robot)
    )
    return grasp


def main():

    pu.connect(use_gui=True)
    pu.set_default_camera(distance=2)
    pu.disable_real_time()
    pu.draw_global_system()
    with pu.HideOutput():
        robot = pu.load_model(pu.DRAKE_IIWA_URDF) # KUKA_IIWA_URDF | DRAKE_IIWA_URDF
        floor = pu.load_model('models/short_floor.urdf')

    mug = pu.load_model("../data/mugs/0.urdf", fixed_base=False)
    pu.set_pose(mug, pu.Pose(pu.Point(y=0.5, z=pu.stable_z(mug, floor))))

    tree = pu.load_model("../data/mug_tree/mug_tree.urdf")
    pu.set_pose(tree, pu.Pose(pu.Point(x=0.5, z=pu.stable_z(tree, floor))))

    pu.dump_world()
    saved_world = pu.WorldSaver()

    ik_fn = kuka_primitives.get_ik_fn(robot, fixed=[floor, tree], teleport=False)
    free_motion_fn = kuka_primitives.get_free_motion_gen(
        robot, fixed=([floor, tree, mug]), teleport=False)
    holding_motion_fn = kuka_primitives.get_holding_motion_gen(
        robot, fixed=[floor, tree], teleport=False)

    pose0 = kuka_primitives.BodyPose(mug)
    conf0 = kuka_primitives.BodyConf(robot)

    grasp = get_mug_grasp(mug, robot)
    result1 = ik_fn(mug, pose0, grasp)
    conf1, path2 = result1
    pose0.assign()

    result2 = free_motion_fn(conf0, conf1)
    path1, = result2

    result3 = holding_motion_fn(conf1, conf0, mug, grasp)
    path3, = result3

    command = kuka_primitives.Command(path1.body_paths + path2.body_paths + path3.body_paths)

    saved_world.restore()
    pu.update_state()

    command.refine(num_steps=10).execute(time_step=0.005)
    pu.wait_if_gui()
    pu.disconnect()


main()
