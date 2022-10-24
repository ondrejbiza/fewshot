import argparse
import pickle
import numpy as np
import open3d as o3d

from pybullet_planning.pybullet_tools import utils as pu
import utils


def show_scene(point_clouds, background=None):

    colors = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=np.float32)

    points = []
    point_colors = []

    for i, key in enumerate(sorted(point_clouds.keys())):
        points.append(point_clouds[key])
        point_colors.append(np.tile(colors[i][None, :], (len(points[-1]), 1)))

    points = np.concatenate(points, axis=0).astype(np.float32)
    point_colors = np.concatenate(point_colors, axis=0)

    if background is not None:
        points = np.concatenate([points, background], axis=0)
        background_colors = np.zeros_like(background)
        background_colors[:] = 0.9
        point_colors = np.concatenate([point_colors, background_colors], axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    utils.o3d_visualize(pcd)


def main(args):

    pu.connect(use_gui=True, show_sliders=False)
    pu.set_default_camera(distance=2)
    pu.disable_real_time()
    pu.draw_global_system()

    with pu.HideOutput():
        floor = pu.load_model('models/short_floor.urdf')
        mug = pu.load_model("../data/mugs/test/{:d}.urdf".format(args.mug_index))
        tree = pu.load_model("../data/simple_trees/test/{:d}.urdf".format(args.tree_index))

    pu.set_pose(mug, pu.Pose(pu.Point(x=0.2, y=0.0, z=pu.stable_z(mug, floor)), pu.Euler(0., 0., 0.)))
    pu.set_pose(tree, pu.Pose(pu.Point(x=-0.0, y=0.0, z=pu.stable_z(tree, floor))))

    canon = {}
    with open("data/mugs_pca.pkl", "rb") as f:
        canon[1] = pickle.load(f)
    with open("data/simple_trees_pca.pkl", "rb") as f:
        canon[2] = pickle.load(f)

    pcs, _ = utils.observe_point_cloud(utils.RealSenseD415.CONFIG, [1, 2])
    
    filled_pcs = {}
    filled_pcs[1] = utils.planar_pose_warp_gd(canon[1]["pca"], canon[1]["canonical_obj"], pcs[1])[0]
    filled_pcs[2] = utils.planar_pose_warp_gd(canon[2]["pca"], canon[2]["canonical_obj"], pcs[2], n_angles=1, object_size_reg=0.1)[0]

    show_scene(filled_pcs, background=np.concatenate(list(pcs.values())))

    pu.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mug-index", type=int, default=0)
    parser.add_argument("-t", "--tree-index", type=int, default=0)
    main(parser.parse_args())
