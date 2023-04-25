import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave, imread
import torch
import pybullet as pb
from pyquaternion import Quaternion

from src import utils, viz_utils
from src.object_warping import ObjectWarpingSE2Batch, ObjectWarpingSE3Batch, warp_to_pcd_se2, warp_to_pcd_se3, PARAM_1
from src.real_world import constants
from src.real_world.simulation import Simulation


def cvK2BulletP(K, w, h, near, far):
    """
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
    A = (near + far)/(near - far)
    B = 2 * near * far / (near - far)

    projection_matrix = [
                        [2/w * f_x,  0,          pp_x,  0],
                        [0,          2/h * f_y,  pp_y,  0],
                        [0,          0,          A,              B],
                        [0,          0,          -1,             0]]
    #The transpose is needed for respecting the array structure of the OpenGL
    return np.array(projection_matrix).T.reshape(16).tolist()


def cvPose2BulletView(q, t):
    """
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
    T=Tc@np.linalg.inv(T)
    # The transpose is needed for respecting the array structure of the OpenGL
    viewMatrix = T.T.reshape(16)
    return viewMatrix


def d435_intrinsics():
    # TODO: get a real camera matrix from D435.

    width = 640
    height = 480

    # My guess.
    # fx = 608.6419
    # fy = 607.1061
    # ppx = width / 2
    # ppy = height / 2

    # Color.
    fx = 604.364
    fy = 603.84
    ppx = 329.917
    ppy = 240.609

    # Depth.
    # fx = 381.814
    # fy = 381.814
    # ppx = 317.193
    # ppy = 239.334

    proj = np.array([
        [fx, 0, ppx],
        [0, fy, ppy],
        [0, 0, 1]
    ], dtype=np.float32)
    return proj, width, height


def depth_to_point_cloud(K, D, depth_scale=1):
    height, width = D.shape

    # Invert the camera matrix
    K_inv = np.linalg.inv(K)

    # Create meshgrid for pixel coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Normalize the pixel coordinates
    normalized_pixel_coords = np.stack([x, y, np.ones_like(x)], axis=-1)

    # Convert depth image to meters
    depth_meters = D * depth_scale / 1000

    # Multiply the normalized pixel coordinates by the inverse camera matrix
    camera_coords = np.matmul(normalized_pixel_coords * depth_meters[..., np.newaxis], K_inv.T)

    # Create the point cloud by reshaping the camera coordinates
    point_cloud = camera_coords.reshape(-1, 3)

    return point_cloud


def main(args):

    classes = ['cup', 'bowl', 'mug', 'bottle', 'cardboard', 'box', 'Tripod', 'Baseball bat' , 'Lamp', 'Mug Rack', 'Plate', 'Toaster', 'Spoon']

    with open(args.save_file, "rb") as f:
        data = pickle.load(f)

    with open(args.save_file2, "rb") as f:
        data2 = pickle.load(f)

    i = 1

    print(data[i].keys())
    pcd = data[i]["cloud"].reshape(480, 640, 3)
    image = data[i]["image"][:, :, ::-1]  # BGR to RGB
    depth = data[i]["depth"]
    masks = data[i]["masks"][:, 0].cpu().numpy()
    class_idx = data[i]["class_idx"]
    texcoords = data2[i]["texcoords"].reshape(480 * 640, 2)
    texcoords[:, 1] = np.clip(texcoords[:, 1] * 480, 0, 480 - 1)
    texcoords[:, 0] = np.clip(texcoords[:, 0] * 640, 0, 640 - 1)
    texcoords = texcoords.astype(np.int32)

    print(image.shape)

    # plt.imshow(image)
    # plt.show()

    pcd = pcd.reshape(480 * 640, 3)
    # viz_utils.show_pcd_plotly(pcd, center=True)

    d = {}
    for j in range(len(class_idx)):
    # for j in [0, 2]:
        name = f"{j}_{classes[class_idx[j]]}"
        mask = masks[j]
        mask2 = mask[texcoords[:, 1], texcoords[:, 0]]
        tmp = pcd[mask2]
        tmp = tmp[np.sqrt(np.sum(np.square(tmp), axis=-1)) < 0.5]
        d[name] = tmp
        print(name)
        # viz_utils.show_pcd_plotly(tmp, center=True)

    # viz_utils.show_pcds_plotly(d, center=True)

    canon_path = constants.NDF_MUGS_PCA_PATH
    canon_scale = constants.NDF_MUGS_INIT_SCALE
    canon = utils.CanonObj.from_pickle(canon_path)

    complete_pcds = {}
    for j in [0, 2]:

        name = f"{j}_{classes[class_idx[j]]}"
        mask = masks[j]
        mask2 = mask[texcoords[:, 1], texcoords[:, 0]]
        tmp = pcd[mask2]
        tmp = tmp[np.sqrt(np.sum(np.square(tmp), axis=-1)) < 0.5]

        if len(tmp) > 2000:
            tmp = utils.farthest_point_sample(tmp, 2000)[0]

        warp = ObjectWarpingSE2Batch(
            canon, tmp, torch.device("cuda:0"), **PARAM_1,
            init_scale=canon_scale)
        complete_pcd, _, param = warp_to_pcd_se2(warp, n_angles=12, n_batches=1)
        complete_pcds[j] = complete_pcd

    # viz_utils.show_pcds_plotly(complete_pcds, center=False)

    with open(args.pick_demo, "rb") as f:
        d = pickle.load(f)
    index = d["index"]
    pos_robotiq = d["pos_robotiq"]
    trans_pre_t0_to_t0 = d["trans_pre_t0_to_t0"]

    source_pcd_complete = canon.to_pcd(param)
    source_points = source_pcd_complete[index]

    # Pick pose in canonical frame..
    trans, _, _ = utils.best_fit_transform(pos_robotiq, source_points)
    # Pick pose in workspace frame.
    trans_robotiq_to_ws = param.get_transform() @ trans

    sim = Simulation()

    source_mesh = canon.to_mesh(param)
    source_mesh.export("tmp_source.stl")
    utils.convex_decomposition(source_mesh, "tmp_source.obj")
    obj_id = sim.add_object("tmp_source.urdf", param.position, param.quat)
    robotiq_id = sim.add_object("data/robotiq.urdf", *utils.transform_to_pos_quat(trans_robotiq_to_ws))

    with open(args.calibration, "rb") as f:
        d = pickle.load(f)

    # This doesn't work.
    # I can use getDebugVisualizerCamera to get the default view matrix and debug the projection matrix.

    cam2world = d["cam2world"]
    pos, quat = utils.transform_to_pos_quat(cam2world)
    view_matrix = cvPose2BulletView(quat, pos)
    proj_matrix = cvK2BulletP(d435_intrinsics()[0], 640, 480, 0.1, 100)

    image_arr = pb.getCameraImage(
        640, 480, viewMatrix=view_matrix, projectionMatrix=proj_matrix
    )
    rgb = image_arr[2]
    print(rgb.shape)

    plt.imshow(rgb)
    plt.show()


# python -m scripts.real_world.experimental.show_dataset2 data/ignore/ondrejs_desk_1_results.pkl data/ignore/ondrejs_desk_1.pkl data/230330/mug_tree_pick.pkl data/ignore/calibration.pickle
parser = argparse.ArgumentParser("Find objects for a particular task.")
parser.add_argument("save_file")
parser.add_argument("save_file2")
parser.add_argument("pick_demo")
parser.add_argument("calibration")
main(parser.parse_args())
