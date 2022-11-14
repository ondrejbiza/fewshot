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


def get_knn_and_deltas(obj, vps):

    k = 10
    # [n_pairs, n_points, 3]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(obj[:, 0], obj[:, 1], obj[:, 2], color="red", alpha=0.1)
    ax.scatter(vps[:, 0], vps[:, 1], vps[:, 2], color="green")
    plt.show()

    dists = np.sum(np.square(obj[None] - vps[:, None]), axis=-1)
    knn_list = []
    deltas_list = []

    for i in range(dists.shape[0]):
        knn = np.argpartition(dists[i], k)[:k]
        deltas = vps[i: i + 1] - obj[knn]
        knn_list.append(knn)
        deltas_list.append(deltas)

    knn_list = np.stack(knn_list)
    deltas_list = np.stack(deltas_list)
    return knn_list, deltas_list


def record_place(mug, tree):

    pos = pu.get_pose(mug)[0]

    dbg = dict()
    dbg['target_x'] = pb.addUserDebugParameter('target_x', WORKSPACE_LOW[0], WORKSPACE_HIGH[0], pos[0])
    dbg['target_y'] = pb.addUserDebugParameter('target_y', WORKSPACE_LOW[1], WORKSPACE_HIGH[1], pos[1])
    dbg['target_z'] = pb.addUserDebugParameter('target_z', -0.5, 0.5, pos[2])
    dbg['save'] =  pb.addUserDebugParameter('save', 1, 0, 0)

    spheres = []

    i = 0
    cols = None
    while True:

        p = utils.read_parameters(dbg)
        pu.set_pose(mug, pu.Pose(pu.Point(p['target_x'], p['target_y'], p['target_z'])))

        if i > 5e+5:  # Update once in a while.
            i = 0
            pb.performCollisionDetection()

            # cols = pb.getContactPoints(mug, tree)
            cols = pb.getClosestPoints(mug, tree, 0.01)

            for s in spheres:
                pu.remove_body(s)
            spheres = []

            for col in cols:
                pos = col[6]
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

    return pos_1, pos_2


def main(args):

    pu.connect(use_gui=True, show_sliders=True)
    pu.set_default_camera(distance=2)
    pu.disable_real_time()
    pu.draw_global_system()

    with pu.HideOutput():
        floor = pu.load_model("models/short_floor.urdf", fixed_base=True)
        robot = pu.load_model(FRANKA_URDF, fixed_base=True)

        mug = pu.load_model("../data/mugs/test/0.urdf")
        pu.set_pose(mug, pu.Pose(pu.Point(x=0.5, y=-0.25, z=pu.stable_z(mug, floor))))

        tree = pu.load_model("../data/simple_trees/test/0.urdf", fixed_base=True)
        pu.set_pose(tree, pu.Pose(pu.Point(x=0.5, y=0.25, z=pu.stable_z(tree, floor))))

    pos_1, pos_2 = record_place(mug, tree)

    canon = {}
    with open("data/mugs_pca.pkl", "rb") as f:
        canon[mug] = pickle.load(f)
    with open("data/simple_trees_pca.pkl", "rb") as f:
        canon[tree] = pickle.load(f)

    pcs, _ = utils.observe_point_cloud(utils.RealSenseD415.CONFIG, [mug, tree])
    
    filled_pcs = {}
    filled_pcs[mug], _, param_1 = utils.planar_pose_warp_gd(canon[mug]["pca"], canon[mug]["canonical_obj"], pcs[mug])
    filled_pcs[tree], _, param_2 = utils.planar_pose_warp_gd(canon[tree]["pca"], canon[tree]["canonical_obj"], pcs[tree], n_angles=1, object_size_reg=0.1)
    viz_utils.show_scene(filled_pcs, background=np.concatenate(list(pcs.values())))

    T1 = np.concatenate([utils.yaw_to_rot(param_1[2]), param_1[1][:, None]], axis=1)
    T1 = np.concatenate([T1, np.array([[0., 0., 0., 1.]])], axis=0)

    vp = pos_2

    knns, deltas = get_knn_and_deltas(filled_pcs[mug], vp)

    tmp = np.concatenate([pos_2, np.ones_like(pos_2)[:, 0: 1]], axis=1)
    vp = np.matmul(np.linalg.inv(T1), tmp.T).T
    vp /= vp[:, -1:]
    vp = vp[:, :-1]

    dist_2 = np.sqrt(np.sum(np.square(filled_pcs[tree][:, None] - pos_2[None]), axis=2))

    i_2 = np.argmin(dist_2, axis=0).transpose()

    with open(args.save_path, "wb") as f:
        pickle.dump({
            "knns": knns,
            "deltas": deltas,
            "target_indices": i_2
        }, f)

    pu.disconnect()


parser = argparse.ArgumentParser()
parser.add_argument("save_path")
main(parser.parse_args())
