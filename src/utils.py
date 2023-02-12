from dataclasses import dataclass
import pickle
from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
import trimesh


@dataclass
class ObjParam:
    """Object shape and pose parameters.
    """
    position: NDArray[np.float64]
    quat: NDArray[np.float64]
    latent: Optional[NDArray[np.float32]] = None

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

    def to_pcd(self, obj_param: ObjParam) -> NDArray[np.float32]:
        if self.pca is not None:
            return self.canonical_pcd + self.pca.inverse_transform(obj_param.latent).reshape(-1, 3)
        else:
            return np.copy(self.canonical_pcd)

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
