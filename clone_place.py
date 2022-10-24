import argparse
import time
import os
import pickle
import numpy as np
import pybullet as pb
from pybullet_planning.pybullet_tools import utils as pu
import utils


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
        mug = pu.load_model("../data/mugs/test/0.urdf")
        tree = pu.load_model("../data/trees/test/0.urdf")

    pu.set_pose(mug, pu.Pose(pu.Point(x=0.2, y=0.0, z=pu.stable_z(mug, floor)), pu.Euler(0., 0., 0.)))
    pu.set_pose(tree, pu.Pose(pu.Point(x=0.0, y=0.0, z=pu.stable_z(tree, floor))))

    pos = pu.get_pose(mug)[0]

    vmin, vmax = -0.2, 0.2
    dbg = dict()
    dbg['target_x'] = pb.addUserDebugParameter('target_x', vmin, vmax, pos[0])
    dbg['target_y'] = pb.addUserDebugParameter('target_y', vmin, vmax, pos[1])
    dbg['target_z'] = pb.addUserDebugParameter('target_z', -0.5, 0.5, pos[2])
    dbg['save'] =  pb.addUserDebugParameter('save', 1, 0, 0)

    spheres = []

    i = 0
    cols = None
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
                with pu.HideOutput():
                    s = pu.load_model("../data/sphere.urdf")
                    pu.set_pose(s, pu.Pose(pu.Point(*pos)))
                spheres.append(s)

        i += 1

        if p['save'] > 0 and cols is not None:
            break

    pos_1 = [col[5] for col in cols]
    pos_2 = [col[6] for col in cols]

    pos_1 = np.stack(pos_1, axis=0).astype(np.float32)
    pos_2 = np.stack(pos_2, axis=0).astype(np.float32)

    canon = {}
    with open("data/mugs_pca.pkl", "rb") as f:
        canon[1] = pickle.load(f)
    with open("data/trees_pca_8d.pkl", "rb") as f:
        canon[2] = pickle.load(f)

    pcs, _ = utils.observe_point_cloud(utils.RealSenseD415.CONFIG, [1, 2])
    
    filled_pcs = {}
    filled_pcs[1] = utils.planar_pose_warp_gd(canon[1]["pca"], canon[1]["canonical_obj"], pcs[1])[0]
    filled_pcs[2] = utils.planar_pose_warp_gd(canon[2]["pca"], canon[2]["canonical_obj"], pcs[2], n_angles=1)[0]

    dist_1 = np.sqrt(np.sum(np.square(filled_pcs[1][:, None] - pos_1[None]), axis=2))
    dist_2 = np.sqrt(np.sum(np.square(filled_pcs[2][:, None] - pos_2[None]), axis=2))

    i_1 = np.argmin(dist_1, axis=0).transpose()
    i_2 = np.argmin(dist_2, axis=0).transpose()

    # i_1 for mug contacts, i_2 for tree contacts
    indices = np.stack([i_1, i_2], axis=0)

    print("Save shape:", indices.shape)
    print(indices)
    np.save(args.save_path, indices)

    pu.disconnect()


parser = argparse.ArgumentParser()
parser.add_argument("save_path")
main(parser.parse_args())
