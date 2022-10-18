import argparse
import time
import os
import numpy as np
import pybullet as pb
import open3d as o3d
from pybullet_planning.pybullet_tools import kuka_primitives
from pybullet_planning.pybullet_tools import utils as pu
import utils
import viz_utils


def main(args):

    pu.connect(use_gui=True, show_sliders=False)
    pu.set_default_camera(distance=2)
    pu.disable_real_time()
    pu.draw_global_system()

    with pu.HideOutput():
        floor = pu.load_model('models/short_floor.urdf')
        mug = pu.load_model("../data/mugs/test/0.urdf")
        tree = pu.load_model("../data/trees/test/0.urdf")

    pu.set_pose(mug, pu.Pose(pu.Point(x=0.2, y=0.0, z=pu.stable_z(mug, floor)), pu.Euler(0., 0., 0.)))
    pu.set_pose(tree, pu.Pose(pu.Point(x=-0.0, y=0.0, z=pu.stable_z(tree, floor))))

    c, d, s = [], [], []
    cfgs = utils.RealSenseD415.CONFIG
    for cfg in cfgs:
        out = utils.render_camera(cfg)
        c.append(out[0])
        d.append(out[1])
        s.append(out[2])

    pcs, colors = utils.reconstruct_segmented_point_cloud(c, d, s, cfgs, [1, 2])
    viz_utils.show_pointcloud(pcs, colors)

    pu.disconnect()


parser = argparse.ArgumentParser()
main(parser.parse_args())
