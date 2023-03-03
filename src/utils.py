from dataclasses import dataclass
import pickle
from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
import pybullet as pb
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
import trimesh

from src import exceptions


@dataclass
class ObjParam:
    """Object shape and pose parameters.
    """
    position: NDArray[np.float64] = np.array([0., 0., 0.])
    quat: NDArray[np.float64] = np.array([0., 0., 0., 1.])
    latent: Optional[NDArray[np.float32]] = None
    scale: NDArray = np.array([1., 1., 1.])

    def get_transform(self):
        return pos_quat_to_transform(self.position, self.quat)


@dataclass
class CanonObj:
    """Canonical object with shape warping.
    """
    canonical_pcd: NDArray[np.float32]
    mesh_vertices: NDArray[np.float32]
    mesh_faces: NDArray[np.float32]
    pca: Optional[PCA] = None

    def __post_init__(self):
        self.n_components = self.pca.n_components

    def to_pcd(self, obj_param: ObjParam) -> NDArray[np.float32]:
        if self.pca is not None and obj_param.latent is not None:
            pcd = self.canonical_pcd + self.pca.inverse_transform(obj_param.latent).reshape(-1, 3)
        else:
            if self.pca is not None:
                print("WARNING: Skipping warping because we do not have a latent vector. We however have PCA.")
            pcd = np.copy(self.canonical_pcd)
        return pcd * obj_param.scale[None]

    def to_transformed_pcd(self, obj_param: ObjParam) -> NDArray[np.float32]:
        pcd = self.to_pcd(obj_param)
        trans = pos_quat_to_transform(obj_param.position, obj_param.quat)
        return transform_pcd(pcd, trans)

    def to_mesh(self, obj_param: ObjParam) -> trimesh.base.Trimesh:
        pcd = self.to_pcd(obj_param)
        # The vertices are assumed to be at the start of the canonical_pcd.
        vertices = pcd[:len(self.mesh_vertices)]
        return trimesh.base.Trimesh(vertices, self.mesh_faces)

    @staticmethod
    def from_pickle(load_path: str) -> "CanonObj":
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        pcd = data["canonical_obj"]
        pca = None
        if "pca" in data:
            pca = data["pca"]
        mesh_vertices = data["canonical_mesh_points"]
        mesh_faces = data["canonical_mesh_faces"]
        return CanonObj(pcd, mesh_vertices, mesh_faces, pca)


@dataclass
class PickDemoSingleVertex:
    """Pick demonstrating with a single canonical object index.
    """
    target_index: int
    target_quat: NDArray[np.float64]


@dataclass
class PickDemoContactPoints:
    """Pick demonstrating with contact points between object and gripper.
    """
    gripper_indices: NDArray[np.int32]
    target_indices: NDArray[np.int32]

    def check_consistent(self):
        assert len(self.gripper_indices) == len(self.target_indices)


@dataclass
class PlaceDemoVirtualPoints:
    """Place demonstration with virtual points.
    """
    knns: NDArray[np.int32]
    deltas: NDArray[np.float64]
    target_indices: NDArray[np.int32]

    def check_consistent(self):
        assert len(self.knns) == len(self.deltas) == len(self.target_indices)


@dataclass
class PlaceDemoContactPoints:
    """Place demonstration with contact points.
    """
    source_indices: NDArray[np.int32]
    target_indices: NDArray[np.int32]

    def check_consistent(self):
        assert len(self.source_indices) == len(self.target_indices)


def quat_to_rotm(quat: NDArray[np.float64]) -> NDArray[np.float64]:
    return Rotation.from_quat(quat).as_matrix()


def rotm_to_quat(rotm: NDArray[np.float64]) -> NDArray[np.float64]:
    return Rotation.from_matrix(rotm).as_quat()


def pos_quat_to_transform(
        pos: Union[Tuple[float, float, float], NDArray[np.float64]],
        quat: Union[Tuple[float, float, float, float], NDArray[np.float64]]
    ) -> NDArray[np.float64]:
    trans = np.eye(4).astype(np.float64)
    trans[:3, 3] = pos
    trans[:3, :3] = quat_to_rotm(np.array(quat))
    return trans


def transform_to_pos_quat(
    trans: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    pos = trans[:3, 3]
    quat = rotm_to_quat(trans[:3, :3])
    return pos, quat


def transform_pcd(
        pcd: NDArray[np.float32], trans: NDArray[np.float64], is_position: bool=True
    ) -> NDArray[np.float32]:
    n = pcd.shape[0]
    cloud = pcd.T
    augment = np.ones((1, n)) if is_position else np.zeros((1, n))
    cloud = np.concatenate((cloud, augment), axis=0)
    cloud = np.dot(trans.astype(np.float32), cloud)
    cloud = cloud[0: 3, :].T
    return cloud


def best_fit_transform(A: NDArray, B: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
    '''
    https://github.com/ClayFlannigan/icp/blob/master/icp.py
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''
    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def convex_decomposition(mesh: trimesh.base.Trimesh, save_path: Optional[str]=None) -> List[trimesh.base.Trimesh]:
    """Convex decomposition of a mesh using testVHCAD through trimesh."""
    convex_meshes = trimesh.decomposition.convex_decomposition(
        mesh, resolution=1000000, depth=20, concavity=0.0025, planeDownsampling=4, convexhullDownsampling=4,
        alpha=0.05, beta=0.05, gamma=0.00125, pca=0, mode=0, maxNumVerticesPerCH=256, minVolumePerCH=0.0001,
        convexhullApproximation=1, oclDeviceID=0
    )

    if save_path is not None:
        decomposed_scene = trimesh.scene.Scene()
        for i, convex_mesh in enumerate(convex_meshes):
            decomposed_scene.add_geometry(convex_mesh, node_name=f"hull_{i}")
        decomposed_scene.export(save_path, file_type="obj")

    return convex_meshes


def farthest_point_sample(point: NDArray, npoint: int) -> Tuple[NDArray, NDArray]:
    # https://github.com/yanx27/Pointnet_Pointnet2_pytorch
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    indices = centroids.astype(np.int32)
    point = point[indices]
    return point, indices


def pb_set_pose(body: int, pos: NDArray, quat: NDArray, sim_id: Optional[int]=None):
    if sim_id is not None:
        pb.resetBasePositionAndOrientation(body, pos, quat, physicsClientId=sim_id)
    else:
        pb.resetBasePositionAndOrientation(body, pos, quat)


def pb_get_pose(body, sim_id: Optional[int]=None) -> Tuple[NDArray, NDArray]:
    if sim_id is not None:
        pos, quat = pb.getBasePositionAndOrientation(body, physicsClientId=sim_id)
    else:
        pos, quat = pb.getBasePositionAndOrientation(body)
    pos = np.array(pos, dtype=np.float64)
    quat = np.array(quat, dtype=np.float64)
    return pos, quat


def pb_body_collision(body1: int, body2: int, sim_id: Optional[int]=None) -> bool:
    if sim_id is not None:
        results = pb.getClosestPoints(bodyA=body1, bodyB=body2, distance=0.0, physicsClientId=sim_id)
    else:
        results = pb.getClosestPoints(bodyA=body1, bodyB=body2, distance=0.0)
    return len(results) != 0


def pb_set_joint_positions(body, joints: List[int], positions: List[float]):
    assert len(joints) == len(positions)
    for joint, position in zip(joints, positions):
        pb.resetJointState(body, joint, targetValue=position, targetVelocity=0)


def wiggle(source_obj: int, target_obj: int, max_tries: int=100000,
           sim_id: Optional[int]=None) -> Tuple[NDArray, NDArray]:
    """Wiggle the source object out of a collision with the target object.
    
    Important: this function will change the state of the world and we assume
    the world was saved before and will be restored after.
    """
    i = 0
    pos, quat = pb_get_pose(source_obj, sim_id=sim_id)

    pb.performCollisionDetection()
    in_collision = pb_body_collision(source_obj, target_obj, sim_id=sim_id)
    if not in_collision:
        return pos, quat

    while True:

        new_pos = pos + np.random.normal(0, 0.01, 3)
        pb_set_pose(source_obj, new_pos, quat, sim_id=sim_id)

        pb.performCollisionDetection()
        in_collision = pb_body_collision(source_obj, target_obj, sim_id=sim_id)
        if not in_collision:
            return new_pos, quat

        i += 1
        if i > max_tries:
            return pos, quat
