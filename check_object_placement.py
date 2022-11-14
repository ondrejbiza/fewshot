import argparse
from typing import List
import time
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pybullet as pb

from pybullet_planning.pybullet_tools import utils as pu
from pybullet_planning.pybullet_tools.ikfast.franka_panda.ik import PANDA_INFO, FRANKA_URDF
import utils
import viz_utils
import exceptions

WORKSPACE_LOW = np.array([0.25, -0.5, 0.], dtype=np.float32)
# TODO: what to do with the z-axis?
WORKSPACE_HIGH = np.array([0.75, 0.5, 0.28], dtype=np.float32)


def main(args):

    pu.connect(use_gui=True, show_sliders=True)
    pu.set_default_camera(distance=2)
    pu.disable_real_time()
    pu.draw_global_system()

    with pu.HideOutput():
        floor = pu.load_model("models/short_floor.urdf", fixed_base=True)
        robot = pu.load_model(FRANKA_URDF, fixed_base=True)

        try:
            placed = []
            for i in range(5):
                mug = pu.load_model("../data/mugs/test/0.urdf")
                utils.place_object(mug, floor, placed, WORKSPACE_LOW, WORKSPACE_HIGH)
                placed.append(mug)
            print("Setup success.")
        except exceptions.EnvironmentSetupError:
            print("Setup fail.")

    for cfg in utils.RealSenseD415.CONFIG:
        out = utils.render_camera(cfg)
        c = out[0]
        plt.imshow(c)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    main(parser.parse_args())
