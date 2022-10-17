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


def main():

    pu.connect(use_gui=True, show_sliders=True)
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

    pos = pu.get_pose(mug)[0]

    vmin, vmax = -1, 1
    dbg = dict()
    dbg['target_x'] = pb.addUserDebugParameter('target_x', vmin, vmax, pos[0])
    dbg['target_y'] = pb.addUserDebugParameter('target_y', vmin, vmax, pos[1])
    dbg['target_z'] = pb.addUserDebugParameter('target_z', vmin, vmax, pos[2])
    dbg['print'] =  pb.addUserDebugParameter('print params', 1, 0, 1)

    spheres = []

    i = 0
    while True:

        p = read_parameters(dbg)
        pu.set_pose(mug, pu.Pose(pu.Point(p['target_x'], p['target_y'], p['target_z'])))

        if i > 5e+5:
            i = 0
            pb.performCollisionDetection()

            cols = pb.getContactPoints(mug, tree)

            for s in spheres:
                pu.remove_body(s)
            spheres = []

            for col in cols:
                pos = col[5]
                s = pu.load_model("../data/sphere.urdf")
                pu.set_pose(s, pu.Pose(pu.Point(*pos)))
                spheres.append(s)

        i += 1

    pu.wait_if_gui()
    pu.disconnect()


main()
