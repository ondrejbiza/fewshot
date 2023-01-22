import time
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import rospy
from sensor_msgs.msg import CameraInfo
import open3d as o3d
import geometry_msgs.msg
import moveit_msgs.msg
import shape_msgs.msg
import pyassimp
import moveit_commander


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


def mask_workspace(cloud: NDArray, desk_center: Tuple[float, float, float], size: float=0.2) -> NDArray:

    cloud = np.copy(cloud)
    cloud[..., 0] -= desk_center[0]
    cloud[..., 1] -= desk_center[1]
    cloud[..., 2] -= desk_center[2]

    mask = np.logical_and(np.abs(cloud[..., 0]) <= size, np.abs(cloud[..., 1]) <= size)
    mask = np.logical_and(mask, cloud[..., 2] >= 0.)
    mask = np.logical_and(mask, cloud[..., 2] <= 2 * size)

    return cloud[mask]


def find_mug_and_tree(cloud: NDArray) -> Tuple[NDArray, NDArray]:

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)

    labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=10))

    pcs = []
    for label in np.unique(labels):
        if label == -1:
            # Background label?
            continue
        
        pc = cloud[labels == label]
        if np.min(pc[..., 2]) > 0.1:
            # Above ground, probably robot gripper.
            continue

        pcs.append(pc)

    assert len(pcs) == 2, "The world must have exactly two objects."
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
    mask = tree[..., 2] >= 0.03
    tree = tree[mask]

    return mug, tree


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


def load_obj_to_moveit_scene(obj_path: str, pos: NDArray, quat: NDArray,
                             obj_base_name: str, moveit_scene: moveit_commander.PlanningSceneInterface):

    msg = geometry_msgs.msg.PoseStamped()
    msg.header.frame_id = "base_link"
    msg.pose = to_pose_message(pos, quat)

    scene = pyassimp.load(obj_path, file_type="obj")
    scale = [1., 1., 1.]

    if not scene.meshes or len(scene.meshes) == 0:
        raise ValueError("There are no meshes in the file")

    for mesh_idx, assimp_mesh in enumerate(scene.meshes):

        if len(assimp_mesh.faces) == 0:
            raise ValueError("There are no faces in the mesh")

        co = moveit_msgs.msg.CollisionObject()
        co.operation = moveit_msgs.msg.CollisionObject.ADD
        co.id = "{:s}_{:d}".format(obj_base_name, mesh_idx)
        co.header = msg.header
        co.pose = msg.pose

        mesh = shape_msgs.msg.Mesh()
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
            assert False, "Unable to build triangles from mesh due to mesh object structure"

        for vertex in assimp_mesh.vertices:
            point = geometry_msgs.msg.Point()
            point.x = vertex[0] * scale[0]
            point.y = vertex[1] * scale[1]
            point.z = vertex[2] * scale[2]
            mesh.vertices.append(point)

        print("mesh {:d}: {:d} verts, {:d} faces".format(mesh_idx, len(mesh.vertices), len(mesh.triangles)))

        co.meshes = [mesh]
        moveit_scene.add_object(co)

    pyassimp.release(scene)


def load_obj_as_cubes_to_moveit_scene(obj_path: str, pos: NDArray, quat: NDArray,
                                      obj_base_name: str, moveit_scene: moveit_commander.PlanningSceneInterface):

    scene = pyassimp.load(obj_path, file_type="obj")

    if not scene.meshes or len(scene.meshes) == 0:
        raise ValueError("There are no meshes in the file")

    for mesh_idx, assimp_mesh in enumerate(scene.meshes):

        if len(assimp_mesh.faces) == 0:
            raise ValueError("There are no faces in the mesh")

        verts = np.array(assimp_mesh.vertices)
        center = np.mean(verts, axis=0)
        verts = verts - center[None]
        scale = np.max(np.abs(verts), axis=0)

        msg = geometry_msgs.msg.PoseStamped()
        msg.header.frame_id = "base_link"
        # TODO: the yaw of the mug is wrong.
        msg.pose = to_pose_message(pos + center, quat)

        moveit_scene.add_box("{:s}_box_{:d}".format(obj_base_name, mesh_idx), msg, scale)

    pyassimp.release(scene)
