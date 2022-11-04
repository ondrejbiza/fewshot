import argparse
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
import pybullet as pb

from pybullet_planning.pybullet_tools.ikfast.franka_panda.ik import PANDA_INFO, FRANKA_URDF
from pybullet_planning.pybullet_tools import utils as pu
from robot import Panda
import utils


def show_scene(point_clouds, background=None):
    import open3d as o3d
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

    # setup sim and robot
    pu.connect(use_gui=True)
    pu.set_default_camera(distance=2)
    if args.real_time:
        pu.enable_real_time()
    else:
        pu.disable_real_time()
    pu.enable_gravity()
    pu.draw_global_system()

    with pu.LockRenderer():
        with pu.HideOutput():
            floor = pu.load_model("models/short_floor.urdf", fixed_base=True)
            robot_id = pu.load_model(FRANKA_URDF, fixed_base=True)
            pu.assign_link_colors(robot_id, max_colors=3, s=0.5, v=1.)

    robot = Panda(robot=robot_id)

    mug = pu.load_model("../data/mugs/test/0.urdf", fixed_base=False)
    tree = pu.load_model("../data/simple_trees/test/0.urdf", fixed_base=True)
    pu.set_pose(mug, pu.Pose(pu.Point(x=0.4, y=0.3, z=pu.stable_z(mug, floor)), pu.Euler(0., 0., np.pi / 4)))
    pu.set_pose(tree, pu.Pose(pu.Point(x=-0.4, y=0.3, z=pu.stable_z(tree, floor))))
    objects = [mug, tree]

    # get registered points on the warped canonical object
    pick_indices = np.load(args.pick_path)
    place_indices = np.load(args.place_path)
    print("Pick index:", pick_indices)
    print("Place indices:", place_indices)

    # get a point cloud
    pcs, _ = utils.observe_point_cloud(utils.RealSenseD415.CONFIG, [2, 3])

    # load canonical objects
    canon = {}
    with open("data/mugs_pca.pkl", "rb") as f:
        canon[2] = pickle.load(f)
    with open("data/simple_trees_pca.pkl", "rb") as f:
        canon[3] = pickle.load(f)

    # fit canonical objects to observed point clouds
    new_obj_1, _, _ = utils.planar_pose_warp_gd(canon[2]["pca"], canon[2]["canonical_obj"], pcs[2])
    new_obj_2, _, _ = utils.planar_pose_warp_gd(canon[3]["pca"], canon[3]["canonical_obj"], pcs[3], n_angles=1, object_size_reg=0.1)
    filled_pcs = {2: new_obj_1, 3: new_obj_2}
    # show_scene(filled_pcs, background=np.concatenate(list(pcs.values())))

    # get grasp
    position = new_obj_1[pick_indices]
    print("Position:", position)

    pose = pu.Pose(position, pu.Euler(roll=np.pi))

    # pu.wait_if_gui()

    path = robot.plan_grasp(pose, 0.15, objects)
    robot.execute_path(path)
    robot.open_hand()

    # pu.wait_if_gui()

    # path = robot.plan_grasp(pose, 0.1, [mug])
    path = robot.plan_grasp_naive(pose, 0.09)
    robot.execute_path(path)
    robot.close_hand()

    print("Picked object:", robot.get_picked_object(objects))

    # pu.wait_if_gui()

    pose = robot.init_tool_pos, pu.get_link_pose(robot.robot, robot.tool_link)[1]
    path = robot.plan_motion(pose, [])
    robot.execute_path(path)

    print("Picked object:", robot.get_picked_object(objects))

    # pu.wait_if_gui()

    pcs, _ = utils.observe_point_cloud(utils.RealSenseD415.CONFIG, [2, 3])
    new_obj_1, _, _ = utils.planar_pose_warp_gd(canon[2]["pca"], canon[2]["canonical_obj"], pcs[2])

    points_1 = new_obj_1[place_indices[0]]
    points_2 = new_obj_2[place_indices[1]]

    # find best pair-wise fit
    p1_to_p2, _, _ = utils.best_fit_transform(points_1, points_2)
    print("Best fit spatial transform:")
    print(p1_to_p2)

    # create a spatial transformation matrix for the mug
    mug_pos, mug_rot = pu.get_pose(mug)
    mug_pos = np.array(mug_pos, dtype=np.float32)
    mug_rot = pu.matrix_from_quat(mug_rot)
    
    mug_T = np.concatenate([mug_rot, mug_pos[:, None]], axis=1)
    mug_T = np.concatenate([mug_T, np.array([[0., 0., 0., 1.]])], axis=0)

    print("Original mug spatial transform:")
    print(mug_T)

    # get a new mug pose
    # new_mug_T = np.matmul(T, mug_T)
    # print("New mug spatial transform:")
    # print(new_mug_T)

    hand_T = pu.get_link_pose(robot.robot, robot.tool_link)
    hand_T = np.concatenate([pu.matrix_from_quat(hand_T[1]), np.array(hand_T[0], dtype=np.float32)[:, None]], axis=1)
    hand_T = np.concatenate([hand_T, np.array([[0., 0., 0., 1.]])], axis=0)

    # w_t_g^-1 w_t_m
    g_t_m = np.matmul(np.linalg.inv(hand_T), mug_T)
    print("g_t_m:", g_t_m)
    g_t_m_pos = g_t_m[:3, 3]
    g_t_m_quat = pu.quat_from_matrix(g_t_m[:3, :3])

    new_hand_T = np.matmul(p1_to_p2, hand_T)
    print("new_hand_T:", new_hand_T)

    # make pose pybullet compatible
    new_hand_rot = new_hand_T[:3, :3]
    new_hand_pos = new_hand_T[:3, 3]
    new_hand_quat = pu.quat_from_matrix(new_hand_rot)

    new_mug_T = np.matmul(p1_to_p2, mug_T)
    new_mug_rot = new_mug_T[:3, :3]
    new_mug_pos = new_mug_T[:3, 3]
    new_mug_quat = pu.quat_from_matrix(new_mug_rot)

    pose = (new_hand_pos, new_hand_quat)

    save = pu.WorldSaver()
    pu.set_pose(mug, (new_mug_pos, new_mug_quat))
    import time ; time.sleep(5)
    pose = utils.wiggle(mug, tree)
    pu.set_pose(mug, pose)
    print("In collision:", pu.body_collision(mug, tree))
    # import time ; time.sleep(100)
    save.restore()

    mug_attach = pu.Attachment(robot.robot, robot.tool_link, (g_t_m_pos, g_t_m_quat), mug)

    # conf = robot.plan_motion(pose, obstacles=[tree], attachments=[mug_attach])
    conf = robot.ik(pose)
    pu.set_joint_positions(robot.robot, robot.ik_joints, conf)
    mug_attach.assign()

    print("In collision:", pu.body_collision(mug, tree))

    # pu.step_simulation()
    # cols = pb.getContactPoints(mug, tree)
    # print("Num colissions:", len(cols))

    # import time ; time.sleep(100)

    path = robot.plan_motion(pose, obstacles=[tree], attachments=[mug_attach])
    robot.execute_path(path)
    robot.open_hand()

    # pu.set_pose(mug, (new_mug_pos, new_mug_quat))

    pu.wait_if_gui()
    pu.disconnect()


parser = argparse.ArgumentParser()
parser.add_argument("--pick-path", default="data/clone_pick.npy")
parser.add_argument("--place-path", default="data/clone_place.npy")
parser.add_argument("-r", "--real-time", default=False, action="store_true")
main(parser.parse_args())
