import argparse
import os
import pybullet as pb
import time
from pybullet_planning.pybullet_tools import utils as pu


def main(args):

    pu.connect(use_gui=True)
    pu.set_default_camera(distance=2)
    pu.disable_real_time()
    pu.draw_global_system()

    with pu.HideOutput():
        floor = pu.load_model('models/short_floor.urdf')

    path = os.path.join("..", args.path)
    obj = pu.load_model(path, fixed_base=False)
    pu.set_pose(obj, pu.Pose(pu.Point(x=0.1, y=0.2, z=pu.stable_z(obj, floor))))

    num_joints = pb.getNumJoints(obj)
    print("# Num. joints: {:d}".format(num_joints))
    for joint_idx in range(num_joints):
        print("# Joint {:d}".format(joint_idx))
        print("## Joint info:", pb.getJointInfo(obj, joint_idx))
        print("## Joint state:", pb.getJointState(obj, joint_idx))

    for link_idx in range(num_joints):
        print("# Link {:d}".format(link_idx))
        print("## Link state:", pb.getLinkState(obj, link_idx))

    time.sleep(1000)


parser = argparse.ArgumentParser()
parser.add_argument("path")
main(parser.parse_args())
