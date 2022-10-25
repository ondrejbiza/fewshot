import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pybullet as pb
import open3d as o3d
from scipy.special import softmax

from pybullet_planning.pybullet_tools import utils as pu
import utils


def main(args):

    pu.connect(use_gui=True, show_sliders=True)
    pu.set_default_camera(distance=2)
    pu.disable_real_time()
    pu.draw_global_system()

    with pu.HideOutput():
        floor = pu.load_model('models/short_floor.urdf')
        mug = pu.load_model("../data/mugs/test/0.urdf")
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

    while True:

        p = utils.read_parameters(dbg)
        pu.set_pose(point, pu.Pose(pu.Point(p['target_x'], p['target_y'], p['target_z'])))

        if p['save'] > 0:
            break

    pos = np.array(pu.get_pose(point)[0], dtype=np.float32)

    canon = {}
    with open("data/mugs_pca.pkl", "rb") as f:
        canon[1] = pickle.load(f)

    pcs, _ = utils.observe_point_cloud(utils.RealSenseD415.CONFIG, [1])
    
    filled_pcs = {}
    filled_pcs[1] = utils.planar_pose_warp_gd(canon[1]["pca"], canon[1]["canonical_obj"], pcs[1])[0]

    bbox = [[np.min(filled_pcs[1][:, 0]), np.max(filled_pcs[1][:, 0])], 
            [np.min(filled_pcs[1][:, 1]), np.max(filled_pcs[1][:, 1])], 
            [np.min(filled_pcs[1][:, 2]), np.max(filled_pcs[1][:, 2])]]
    max_dist = np.max([bbox[0][1] - bbox[0][0], bbox[1][1] - bbox[1][0], bbox[2][1] - bbox[2][0]])
    dist = np.sqrt(np.sum(np.square(filled_pcs[1] - pos[None, :]), axis=1)) / max_dist

    plt.subplot(1, 2, 1)
    plt.hist(dist)
    
    temp = np.exp(5)
    probs = softmax(- dist * temp, axis=0)

    plt.subplot(1, 2, 2)
    plt.hist(dist)

    plt.show()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filled_pcs[1])
    color = np.zeros((len(probs), 3), dtype=np.float32)
    color[:, 0] = probs
    pcd.colors = o3d.utility.Vector3dVector(color)
    utils.o3d_visualize(pcd)

    np.save(args.save_path, probs)

    pu.disconnect()


parser = argparse.ArgumentParser()
parser.add_argument("save_path")
main(parser.parse_args())
