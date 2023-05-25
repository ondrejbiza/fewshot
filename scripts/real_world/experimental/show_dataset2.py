import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave
import torch
import pybullet as pb
from pyquaternion import Quaternion
import open3d as o3d
from sklearn.cluster import DBSCAN

from src import utils, viz_utils
from src.object_warping import ObjectWarpingSE2Batch, ObjectWarpingSE3Batch, warp_to_pcd_se2, warp_to_pcd_se3, PARAM_1
from src.real_world import constants
from src.real_world.simulation import Simulation


DEFAULT_FX = 604.364
DEFAULT_FY = 603.84
DEFAULT_PPX = 329.917
DEFAULT_PPY = 240.609
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
NUM_SUBSAMPLE_POINTS = 2000


def convert_to_pb_projection_matrix(K, w, h, near, far):
    """
    https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=12901
    cvKtoPulletP converst the K interinsic matrix as calibrated using Opencv
    and ROS to the projection matrix used in openGL and Pybullet.

    :param K:  OpenCV 3x3 camera intrinsic matrix
    :param w:  Image width
    :param h:  Image height
    :near:     The nearest objects to be included in the render
    :far:      The furthest objects to be included in the render
    :return:   4x4 projection matrix as used in openGL and pybullet
    """ 
    f_x = K[0,0]
    f_y = K[1,1]
    pp_x = K[0,2]
    pp_y = K[1,2]
    A = (near + far) / (near - far)
    B = 2 * near * far / (near - far)

    projection_matrix = [
                        [2/w * f_x,  0,          (w - 2*pp_x)/w,  0],
                        [0,          2/h * f_y,  (2*pp_y - h)/h,  0],
                        [0,          0,          A,              B],
                        [0,          0,          -1,             0]]
    #The transpose is needed for respecting the array structure of the OpenGL
    return np.array(projection_matrix).T.reshape(16).tolist()


def convert_to_pb_view_matrix(q, t):
    """
    https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=12901
    cvPose2BulletView gets orientation and position as used 
    in ROS-TF and opencv and coverts it to the view matrix used 
    in openGL and pyBullet.
    
    :param q: ROS orientation expressed as quaternion [qx, qy, qz, qw] 
    :param t: ROS postion expressed as [tx, ty, tz]
    :return:  4x4 view matrix as used in pybullet and openGL
    
    """
    q = Quaternion([q[3], q[0], q[1], q[2]])
    R = q.rotation_matrix

    T = np.vstack([np.hstack([R, np.array(t).reshape(3,1)]),
                              np.array([0, 0, 0, 1])])
    # Convert opencv convention to python convention
    # By a 180 degrees rotation along X
    Tc = np.array([[1,   0,    0,  0],
                   [0,  -1,    0,  0],
                   [0,   0,   -1,  0],
                   [0,   0,    0,  1]]).reshape(4,4)

    # pybullet pse is the inverse of the pose from the ROS-TF
    T = Tc @ np.linalg.inv(T)
    # The transpose is needed for respecting the array structure of the OpenGL
    viewMatrix = T.T.reshape(16)
    return viewMatrix


def get_proj_matrix(fx, fy, ppx, ppy):

    return np.array([
        [fx, 0, ppx],
        [0, fy, ppy],
        [0, 0, 1]
    ], dtype=np.float32)


def postprocess_point_cloud(pcd, mask, texcoords):

    mask2 = mask[texcoords[:, 1], texcoords[:, 0]]
    tmp = pcd[mask2]
    tmp = tmp[np.sqrt(np.sum(np.square(tmp), axis=-1)) < 0.5]

    if len(tmp) > 100:
        labels = DBSCAN(eps=0.03, min_samples=10).fit_predict(tmp)
        pcds = []
        for label in np.unique(labels):
            pcds.append(tmp[labels == label])
        tmp = pcds[np.argmax([len(tmp2) for tmp2 in pcds])]

    if len(tmp) > NUM_SUBSAMPLE_POINTS:
        tmp = utils.farthest_point_sample(tmp, NUM_SUBSAMPLE_POINTS)[0]
    return tmp


def warp_to_pcd(canon, pcd, canon_scale):

    warp = ObjectWarpingSE2Batch(
        canon, pcd, torch.device("cuda:0"), **PARAM_1,
        init_scale=canon_scale)
    complete_pcd, _, param = warp_to_pcd_se2(warp, n_angles=12, n_batches=1)
    return complete_pcd, param


def take_sim_image(cam2world, proj):
    pos, quat = utils.transform_to_pos_quat(cam2world)
    view_matrix = convert_to_pb_view_matrix(quat, pos)
    proj_matrix = convert_to_pb_projection_matrix(
        proj, DEFAULT_WIDTH, DEFAULT_HEIGHT, 0.01, 100)

    image_arr = pb.getCameraImage(
        DEFAULT_WIDTH, DEFAULT_HEIGHT, viewMatrix=view_matrix, projectionMatrix=proj_matrix
    )
    rgb = image_arr[2][:, :, :3]
    return rgb


def project_sim_image(orig_image, sim_image):
    orig_image = np.copy(orig_image)
    mask = np.any(sim_image != 255, axis=-1)
    orig_image[mask] = (0.1 * orig_image[mask] + 0.9 * sim_image[mask]).astype(np.uint8)
    return orig_image


def main(args):

    i = args.image_index

    # List of classes given to segment anything.
    classes = ["cup", "bowl", "mug", "bottle", "cardboard", "box", "Tripod", "Baseball bat",
               "Lamp", "Mug Rack", "Plate", "Toaster", "Spoon"]

    with open(args.save_file, "rb") as f:
        data = pickle.load(f)

    pcd = data[i]["clouds"].reshape(DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)
    image = data[i]["images"][:, :, ::-1]  # BGR to RGB
    masks = data[i]["masks"][:, 0].cpu().numpy()
    class_idx = data[i]["class_idx"]
    texcoords = data[i]["texcoords"].reshape(DEFAULT_HEIGHT * DEFAULT_WIDTH, 2)
    texcoords[:, 1] = np.clip(texcoords[:, 1] * DEFAULT_HEIGHT, 0, DEFAULT_HEIGHT - 1)
    texcoords[:, 0] = np.clip(texcoords[:, 0] * DEFAULT_WIDTH, 0, DEFAULT_WIDTH - 1)
    texcoords = texcoords.astype(np.int32)
    cam2world = data[i]["cam2world"]

    proj = get_proj_matrix(data[i]["color_fx"], data[i]["color_fy"], data[i]["color_ppx"], data[i]["color_ppy"])

    imsave("1.png", image)

    pcd = pcd.reshape(DEFAULT_HEIGHT * DEFAULT_WIDTH, 3)

    d = {}

    if args.j_index is not None:
        r = args.j_index
    else:
        r = range(len(class_idx))

    pcds = []
    pcd_classes = []

    for j in r:
        tmp = postprocess_point_cloud(pcd, masks[j], texcoords)
        if len(tmp) == 0:
            continue

        name = f"{j}_{classes[class_idx[j]]}"
        d[name] = tmp
        print(name)

        pcds.append(tmp)
        pcd_classes.append(class_idx[j])

    if args.viz:
        viz_utils.show_pcds_plotly(d, center=True)

    mug_canon_path = constants.NDF_MUGS_PCA_PATH
    mug_canon_scale = constants.NDF_MUGS_INIT_SCALE
    mug_canon = utils.CanonObj.from_pickle(mug_canon_path)
    mug_demo_path = "data/230330/mug_tree_pick.pkl"

    bottle_canon_path = constants.NDF_BOTTLES_PCA_PATH
    bottle_canon_scale = constants.NDF_BOTTLES_INIT_SCALE
    bottle_canon = utils.CanonObj.from_pickle(bottle_canon_path)
    bottle_demo_path = "data/230330/bottle_in_box_pick.pkl"

    bowl_canon_path = constants.NDF_BOWLS_PCA_PATH
    bowl_canon_scale = constants.NDF_BOWLS_INIT_SCALE
    bowl_canon = utils.CanonObj.from_pickle(bowl_canon_path)
    bowl_demo_path = "data/230330/bowl_on_mug_pick.pkl"

    sim = Simulation(use_gui=True)

    tmp = np.copy(image)
    for jj, j in enumerate(r):
        tmp[masks[j]] = (0.5 * tmp[masks[j]]).astype(np.uint8)
        tmp[masks[j], jj % 3] += 255 // 2

    imsave("2.png", tmp)

    complete_pcds = {}

    complete_pcds = []
    params = []
    for j in range(len(pcds)):
        tmp = pcds[j]

        if pcd_classes[j] == 2:
            # bowl
            canon = mug_canon
            canon_scale = mug_canon_scale
        elif pcd_classes[j] == 3:
            # bottle
            canon = bottle_canon
            canon_scale = bottle_canon_scale
        elif pcd_classes[j] == 1:
            # bowl
            canon = bowl_canon
            canon_scale = bowl_canon_scale
        else:
            raise ValueError("We don't know this class.")

        tmp1, tmp2 = warp_to_pcd(canon, tmp, canon_scale)
        complete_pcds.append(tmp1)
        params.append(tmp2)

    for j in range(len(params)):

        if pcd_classes[j] == 2:
            # bowl
            canon = mug_canon
            canon_scale = mug_canon_scale
        elif pcd_classes[j] == 3:
            # bottle
            canon = bottle_canon
            canon_scale = bottle_canon_scale
        elif pcd_classes[j] == 1:
            # bowl
            canon = bowl_canon
            canon_scale = bowl_canon_scale
        else:
            raise ValueError("We don't know this class.")

        source_mesh = canon.to_mesh(params[j])
        source_mesh.export(f"tmp_source_{j}.stl")
        utils.convex_decomposition(source_mesh, f"tmp_source_{j}.obj")
        obj_id = sim.add_object(f"tmp_source_{j}.urdf", params[j].position, params[j].quat)

        rgbd = np.array([0., 0., 0., 1.])
        rgbd[j % 3] = 0.8
        pb.changeVisualShape(obj_id, -1, rgbaColor=rgbd)

    sim_image = take_sim_image(cam2world, proj)
    tmp_image = project_sim_image(image, sim_image)
    imsave("3.png", tmp_image)
    imsave("3_sim.png", sim_image)

    for j in range(len(params)):

        if pcd_classes[j] == 2:
            # bowl
            canon = mug_canon
            canon_scale = mug_canon_scale
            demo_path = mug_demo_path
        elif pcd_classes[j] == 3:
            # bottle
            canon = bottle_canon
            canon_scale = bottle_canon_scale
            demo_path = bottle_demo_path
        elif pcd_classes[j] == 1:
            # bowl
            canon = bowl_canon
            canon_scale = bowl_canon_scale
            demo_path = bowl_demo_path
        else:
            raise ValueError("We don't know this class.")

        with open(demo_path, "rb") as f:
            d = pickle.load(f)
        index = d["index"]
        pos_robotiq = d["pos_robotiq"]

        source_pcd_complete = canon.to_pcd(params[j])
        source_points = source_pcd_complete[index]

        # Pick pose in canonical frame.
        trans, _, _ = utils.best_fit_transform(pos_robotiq, source_points)
        # Pick pose in workspace frame.
        trans_robotiq_to_ws = params[j].get_transform() @ trans

        robotiq_id = sim.add_object("data/robotiq.urdf", *utils.transform_to_pos_quat(trans_robotiq_to_ws))
        fract = 0.5
        utils.pb_set_joint_positions(robotiq_id, [0, 2, 4, 5, 6, 7], [fract, fract, fract, -fract, fract, -fract])

        for k in list(range(pb.getNumJoints(robotiq_id))) + [-1]:
            rgbd = np.array([0.4, 0.4, 0.4, 1.])
            pb.changeVisualShape(robotiq_id, k, rgbaColor=rgbd)

    sim_image = take_sim_image(cam2world, proj)
    tmp_image = project_sim_image(image, sim_image)
    imsave("4.png", tmp_image)
    imsave("4_sim.png", sim_image)

    if args.viz:
        plt.imshow(tmp_image)
        plt.show()


# python -m scripts.real_world.experimental.show_dataset2 data/ignore/ondrejs_desk_1_results.pkl data/ignore/ondrejs_desk_1.pkl data/230330/mug_tree_pick.pkl data/ignore/calibration.pickle
parser = argparse.ArgumentParser("Find objects for a particular task.")
parser.add_argument("save_file")
parser.add_argument("image_index", type=int, default=0)
parser.add_argument("-j", "--j-index", nargs="+", default=None, type=int)
parser.add_argument("-v", "--viz", default=False, action="store_true")
main(parser.parse_args())
