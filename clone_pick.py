import argparse
import time
import os
import numpy as np
import pybullet as pb
from pybullet_planning.pybullet_tools import kuka_primitives
from pybullet_planning.pybullet_tools import utils as pu


def read_parameters(dbg_params):
    values = dict()
    for name, param in dbg_params.items():
        values[name] = pb.readUserDebugParameter(param)
    return values


def main(args):

    pu.connect(use_gui=True, show_sliders=True)
    pu.set_default_camera(distance=2)
    pu.disable_real_time()
    pu.draw_global_system()

    with pu.HideOutput():
        floor = pu.load_model('models/short_floor.urdf')
        mug = pu.load_model("../data/mugs/0.urdf", fixed_base=False)
        point = pu.load_model("../data/sphere.urdf")

    pu.set_pose(mug, pu.Pose(pu.Point(x=0.0, y=0.0, z=pu.stable_z(mug, floor)), pu.Euler(0., 0., 0.)))
    pu.set_pose(point, pu.Pose(pu.Point(x=0.2, y=0.0, z=pu.stable_z(mug, floor)), pu.Euler(0., 0., 0.)))

    pos = pu.get_pose(point)[0]

    vmin, vmax = -0.2, 0.2
    dbg = dict()
    dbg['target_x'] = pb.addUserDebugParameter('target_x', vmin, vmax, pos[0])
    dbg['target_y'] = pb.addUserDebugParameter('target_y', vmin, vmax, pos[1])
    dbg['target_z'] = pb.addUserDebugParameter('target_z', vmin, vmax, pos[2])
    dbg['save'] =  pb.addUserDebugParameter('save', 1, 0, 0)

    spheres = []

    i = 0
    while True:

        p = read_parameters(dbg)
        pu.set_pose(point, pu.Pose(pu.Point(p['target_x'], p['target_y'], p['target_z'])))

        if p['save'] > 0:
            break

    pos = np.array(pu.get_pose(point)[0], dtype=np.float32)

    print("Save shape:", pos.shape)
    print(pos)
    np.save(args.save_path, pos)

    pu.disconnect()


parser = argparse.ArgumentParser()
parser.add_argument("save_path")
main(parser.parse_args())
