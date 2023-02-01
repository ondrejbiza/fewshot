import time
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import rospy
from sensor_msgs.msg import CameraInfo
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
import open3d as o3d
import geometry_msgs.msg
import moveit_msgs.msg
import shape_msgs.msg
import pyassimp
import moveit_commander
from scipy.spatial.transform import Rotation

from online_isec.tf_proxy import TFProxy
import utils


def get_camera_intrinsics_and_distortion(topic: str) -> Tuple[NDArray, NDArray]:

    out = [False, None, None]
    def callback(msg: CameraInfo):
        out[1] = np.array(msg.K, dtype=np.float64).reshape(3, 3)
        out[2] = np.array(msg.D, dtype=np.float64)
        out[0] = True
    
    sub = rospy.Subscriber(topic, CameraInfo, callback, queue_size=1)
    for _ in range(100):
        time.sleep(0.1)
        if out[0]:
            sub.unregister()
            return out[1], out[2]

    raise RuntimeError("Could not get camera information.")


def mask_workspace(cloud: NDArray, desk_center: Tuple[float, float, float], size: float=0.2, height_eps: float=0.005) -> NDArray:

    cloud = np.copy(cloud)
    cloud[..., 0] -= desk_center[0]
    cloud[..., 1] -= desk_center[1]
    cloud[..., 2] -= desk_center[2]

    mask = np.logical_and(np.abs(cloud[..., 0]) <= size, np.abs(cloud[..., 1]) <= size)
    mask = np.logical_and(mask, cloud[..., 2] >= height_eps)
    mask = np.logical_and(mask, cloud[..., 2] <= 2 * size)

    return cloud[mask]


def find_mug_and_tree(cloud: NDArray) -> Tuple[NDArray, NDArray]:

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)

    labels = np.array(pcd.cluster_dbscan(eps=0.03, min_points=10))

    print("PC lengths (ignoring PCs above the ground).")
    pcs = []
    for label in np.unique(labels):
        if label == -1:
            # Background label?
            continue
        
        pc = cloud[labels == label]
        if np.min(pc[..., 2]) > 0.1:
            # Above ground, probably robot gripper.
            continue

        print(len(pc))

        pcs.append(pc)

    assert len(pcs) >= 2, "The world must have at least two objects."
    if len(pcs) > 2:
        # Pick the two point clouds with the most points.
        sizes = [len(pc) for pc in pcs]
        sort = list(reversed(np.argsort(sizes)))
        pcs = [pcs[sort[0]], pcs[sort[1]]]

    assert len(pcs[0]) > 10, "Too small PC."
    assert len(pcs[1]) > 10, "Too small PC."

    # Tree is taller than mug.
    if np.max(pcs[0][..., 2]) > np.max(pcs[1][..., 2]):
        tree = pcs[0]
        mug = pcs[1]
    else:
        tree = pcs[1]
        mug = pcs[0]
    
    # Cut off the base of the tree.
    # No base.
    mask = tree[..., 2] >= 0.03
    # With base.
    # mask = tree[..., 2] >= 0.05
    tree = tree[mask]

    return mug, tree


def find_held_mug(cloud: NDArray) -> NDArray:

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)

    labels = np.array(pcd.cluster_dbscan(eps=0.03, min_points=10))

    print("PC lengths (ignoring PCs above the ground).")
    pcs = []
    for label in np.unique(labels):
        if label == -1:
            # Background label?
            continue
        
        pc = cloud[labels == label]
        if np.min(pc[..., 2]) > 0.1:
            # Above ground, probably robot gripper.
            continue

        print(len(pc))

        pcs.append(pc)

    sizes = [len(pc) for pc in pcs]
    sort = list(reversed(np.argsort(sizes)))

    return pcs[sort[0]]


def to_stamped_pose_message(pos: NDArray, quat: NDArray, frame_id: str) -> geometry_msgs.msg.PoseStamped:

    msg = geometry_msgs.msg.PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose = to_pose_message(pos, quat)
    return msg


def to_pose_message(pos: NDArray, quat: NDArray) -> geometry_msgs.msg.Pose:

    msg = geometry_msgs.msg.Pose()
    msg.position = to_point_msg(pos.astype(np.float64))
    msg.orientation = to_quat_msg(quat.astype(np.float64))
    return msg


def to_point_msg(pos: NDArray[np.float64]) -> geometry_msgs.msg.Point:

    msg = geometry_msgs.msg.Point()
    msg.x = pos[0]
    msg.y = pos[1]
    msg.z = pos[2]
    return msg


def to_quat_msg(quat: NDArray[np.float64]) -> geometry_msgs.msg.Quaternion:

    msg = geometry_msgs.msg.Quaternion()
    msg.x = quat[0]
    msg.y = quat[1]
    msg.z = quat[2]
    msg.w = quat[3]
    return msg


def load_obj_to_moveit_scene(obj_path: str, pos: NDArray, quat: NDArray, obj_name: str,
                             moveit_scene: moveit_commander.PlanningSceneInterface, file_type: str="obj"):

    scene = pyassimp.load(obj_path, file_type=file_type)
    scale = [1., 1., 1.]

    if not scene.meshes or len(scene.meshes) == 0:
        raise ValueError("There are no meshes in the file")

    meshes = []
    for mesh_idx, assimp_mesh in enumerate(scene.meshes):

        if len(assimp_mesh.faces) == 0:
            raise ValueError("There are no faces in the mesh")

        mesh = shape_msgs.msg.Mesh()
        meshes.append(mesh)

        first_face = assimp_mesh.faces[0]
        if hasattr(first_face, "__len__"):
            for face in assimp_mesh.faces:
                if len(face) == 3:
                    triangle = shape_msgs.msg.MeshTriangle()
                    triangle.vertex_indices = [face[0], face[1], face[2]]
                    mesh.triangles.append(triangle)
        elif hasattr(first_face, "indices"):
            for face in assimp_mesh.faces:
                if len(face.indices) == 3:
                    triangle = shape_msgs.msg.MeshTriangle()
                    triangle.vertex_indices = [
                        face.indices[0],
                        face.indices[1],
                        face.indices[2],
                    ]
                    mesh.triangles.append(triangle)
        else:
            raise ValueError("Unable to build triangles from mesh due to mesh object structure")

        for vertex in assimp_mesh.vertices:
            point = geometry_msgs.msg.Point()
            point.x = vertex[0] * scale[0]
            point.y = vertex[1] * scale[1]
            point.z = vertex[2] * scale[2]
            mesh.vertices.append(point)

        print("mesh {:d}: {:d} verts, {:d} faces".format(mesh_idx, len(mesh.vertices), len(mesh.triangles)))

    msg = to_stamped_pose_message(pos, quat, "base_link")

    co = moveit_msgs.msg.CollisionObject()
    co.operation = moveit_msgs.msg.CollisionObject.ADD
    co.id = obj_name
    co.header = msg.header
    co.pose = msg.pose
    co.meshes = meshes
    moveit_scene.add_object(co)

    pyassimp.release(scene)


def load_obj_to_moveit_scene_2(obj_path: str, pos: NDArray, quat: NDArray, obj_name: str,
                               moveit_scene: moveit_commander.PlanningSceneInterface):

    msg = to_stamped_pose_message(pos, quat, "base_link")
    moveit_scene.add_mesh(obj_name, msg, obj_path)


def load_obj_as_cubes_to_moveit_scene(obj_path: str, pos: NDArray, quat: NDArray,
                                      obj_base_name: str, moveit_scene: moveit_commander.PlanningSceneInterface):

    scene = pyassimp.load(obj_path, file_type="obj")

    if not scene.meshes or len(scene.meshes) == 0:
        raise ValueError("There are no meshes in the file")

    for mesh_idx, assimp_mesh in enumerate(scene.meshes):

        if len(assimp_mesh.faces) == 0:
            raise ValueError("There are no faces in the mesh")

        verts = np.array(assimp_mesh.vertices)
        rot = Rotation.from_quat(quat).as_matrix()
        verts = np.matmul(verts, rot.T)

        center = np.mean(verts, axis=0)
        verts = verts - center[None]
        scale = np.max(np.abs(verts), axis=0) * 2

        msg = to_stamped_pose_message(pos + center, np.array([1., 0., 0., 0.]), "base_link")

        moveit_scene.add_box("{:s}_box_{:d}".format(obj_base_name, mesh_idx), msg, scale)

    pyassimp.release(scene)


def check_added_to_moveit_scene(
    obj_name: str, moveit_scene: moveit_commander.PlanningSceneInterface, timeout: int=2,
    obj_is_known: bool=True, obj_is_attached: bool=False) -> bool:
    """
    Set obj_is_known=False to wait for an object to be deleted.
    Set obj_is_attached=True to wait for an object to be attached to another object.
    """

    start = rospy.get_time()
    seconds = rospy.get_time()
    while (seconds - start < timeout) and not rospy.is_shutdown():
        # Test if the box is in attached objects
        attached_objects = moveit_scene.get_attached_objects([obj_name])
        is_attached = len(attached_objects.keys()) > 0

        # Test if the box is in the scene.
        # Note that attaching the box will remove it from known_objects
        is_known = obj_name in moveit_scene.get_known_object_names()

        # Test if we are in the expected state
        if (obj_is_attached == is_attached) and (obj_is_known == is_known):
            return True

        # Sleep so that we give other threads time on the processor
        rospy.sleep(0.1)
        seconds = rospy.get_time()

    # If we exited the while loop without returning then we timed out
    return False


def check_clean_moveit_scene(
    moveit_scene: moveit_commander.PlanningSceneInterface, timeout: int=2) -> bool:

    start = rospy.get_time()
    seconds = rospy.get_time()
    while (seconds - start < timeout) and not rospy.is_shutdown():

        if len(moveit_scene.get_known_object_names()) == 0:
            return True

        # Sleep so that we give other threads time on the processor
        rospy.sleep(0.1)
        seconds = rospy.get_time()

    # If we exited the while loop without returning then we timed out
    return False


def base_tool0_controller_to_base_link_flange(T: NDArray, tf_proxy: TFProxy) -> NDArray:

    T_b_to_g_prime = T
    T_bl_to_b = np.linalg.inv(tf_proxy.lookup_transform("base_link", "base"))
    T_g_to_b = np.linalg.inv(tf_proxy.lookup_transform("tool0_controller", "base"))
    T_b_to_f = tf_proxy.lookup_transform("flange", "base")
    return T_bl_to_b @ T_b_to_g_prime @ T_g_to_b @ T_b_to_f


def tool0_controller_base_to_flange_base_link(T: NDArray, tf_proxy: TFProxy) -> NDArray:

    T_b_to_bl = tf_proxy.lookup_transform("base", "base_link")
    T_f_to_g = tf_proxy.lookup_transform("flange", "tool0_controller")

    return T_b_to_bl @ T @ T_f_to_g

def desk_obj_param_to_base_link_T(obj_mean: NDArray, obj_yaw: NDArray, desk_center: NDArray,
                                  tf_proxy: TFProxy) -> NDArray:

    T_b_to_m = utils.pos_rot_to_transform(obj_mean + desk_center, utils.yaw_to_rot(obj_yaw))
    T_bl_to_b = np.linalg.inv(tf_proxy.lookup_transform("base_link", "base"))
    return T_bl_to_b @ T_b_to_m


def attach_obj_to_hand(name, moveit_scene):

    # TODO: If the hand is deeper, the object might touch the upper parts of the gripper.
    touch_links = [
        "robotiq_85_base_link",
        "robotiq_85_left_finger_link",
        "robotiq_85_left_finger_tip_link",
        "robotiq_85_left_inner_knuckle_list",
        "robotiq_85_left_knuckle_link",
        "robotiq_85_right_finger_link",
        "robotiq_85_right_finger_tip_link",
        "robotiq_85_right_inner_knuckle_link",
        "robotiq_85_right_knuckle_link"
    ]
    aco = AttachedCollisionObject()
    aco.object = CollisionObject()
    aco.object.id = name
    aco.link_name = "flange"
    aco.touch_links = touch_links
    moveit_scene.attach_object(aco)
