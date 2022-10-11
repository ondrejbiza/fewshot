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

    pb.setPhysicsEngineParameter(enableFileCaching=0)

    while True:

        path = os.path.join("..", args.path)
        obj = pu.load_model(path, fixed_base=False)    
        print("Num. joints: {:d}".format(pb.getNumJoints(obj)))

        time.sleep(1)
        pu.remove_body(obj)


parser = argparse.ArgumentParser()
parser.add_argument("path")
main(parser.parse_args())
