from dataclasses import dataclass
import pickle
from typing import List, Optional, Tuple, Union

from cycpd import deformable_registration
import numpy as np
from numpy.typing import NDArray
import pybullet as pb
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
import trimesh
import trimesh.voxel.creation as vcreate
import torch

from src import exceptions, viz_utils

NPF32 = NDArray[np.float32]
NPF64 = NDArray[np.float64]


@dataclass
class ObjParam:
    """Object shape and pose parameters.
    """
    position: NPF64 = np.array([0., 0., 0.])
    quat: NPF64 = np.array([0., 0., 0., 1.])
    latent: Optional[NPF32] = None
    scale: NPF32 = np.array([1., 1., 1.], dtype=np.float32)

    def get_transform(self) -> NPF64:
        return pos_quat_to_transform(self.position, self.quat)


@dataclass
class CanonObj:
    """Canonical object with shape warping.
    """
    canonical_pcd: NPF32
    mesh_vertices: NPF32
    mesh_faces: NDArray[np.int32]
    pca: Optional[PCA] = None

    def __post_init__(self):
        if self.pca is not None:
            self.n_components = self.pca.n_components

    def to_pcd(self, obj_param: ObjParam) -> NPF32:
        if self.pca is not None and obj_param.latent is not None:
            pcd = self.canonical_pcd + self.pca.inverse_transform(obj_param.latent).reshape(-1, 3)
        else:
            if self.pca is not None:
                print("WARNING: Skipping warping because we do not have a latent vector. We however have PCA.")
            pcd = np.copy(self.canonical_pcd)
        return pcd * obj_param.scale[None]

    def to_transformed_pcd(self, obj_param: ObjParam) -> NPF32:
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
    target_quat: NPF64


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
    deltas: NPF32
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


def quat_to_rotm(quat: NPF64) -> NPF64:
    return Rotation.from_quat(quat).as_matrix()


def rotm_to_quat(rotm: NPF64) -> NPF64:
    return Rotation.from_matrix(rotm).as_quat()


def pos_quat_to_transform(
        pos: Union[Tuple[float, float, float], NPF64],
        quat: Union[Tuple[float, float, float, float], NPF64]
    ) -> NPF64:
    trans = np.eye(4).astype(np.float64)
    trans[:3, 3] = pos
    trans[:3, :3] = quat_to_rotm(np.array(quat))
    return trans


def transform_to_pos_quat(trans: NPF64) -> Tuple[NPF64, NPF64]:
    pos = trans[:3, 3]
    quat = rotm_to_quat(trans[:3, :3])
    # Just making sure.
    return pos.astype(np.float64), quat.astype(np.float64)


def transform_pcd(pcd: NPF32, trans: NPF64, is_position: bool=True) -> NPF32:
    n = pcd.shape[0]
    cloud = pcd.T
    augment = np.ones((1, n)) if is_position else np.zeros((1, n))
    cloud = np.concatenate((cloud, augment), axis=0)
    cloud = np.dot(trans.astype(np.float32), cloud)
    cloud = cloud[0: 3, :].T
    return cloud


def best_fit_transform(A: NPF32, B: NPF32) -> Tuple[NPF64, NPF64, NPF64]:
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
    T = np.identity(m+1).astype(np.float64)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R.astype(np.float64), t.astype(np.float64)


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


def farthest_point_sample(point: NPF32, npoint: int) -> Tuple[NPF32, NDArray[np.int32]]:
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


def pb_set_pose(body: int, pos: NPF64, quat: NPF64, sim_id: Optional[int]=None):
    if sim_id is not None:
        pb.resetBasePositionAndOrientation(body, pos, quat, physicsClientId=sim_id)
    else:
        pb.resetBasePositionAndOrientation(body, pos, quat)


def pb_get_pose(body, sim_id: Optional[int]=None) -> Tuple[NPF64, NPF64]:
    if sim_id is not None:
        pos, quat = pb.getBasePositionAndOrientation(body, physicsClientId=sim_id)
    else:
        pos, quat = pb.getBasePositionAndOrientation(body)
    pos = np.array(pos, dtype=np.float64)
    quat = np.array(quat, dtype=np.float64)
    return pos, quat


def pb_body_collision(body1: int, body2: int, sim_id: Optional[int]=None, margin: float=0.) -> bool:
    if sim_id is not None:
        results = pb.getClosestPoints(bodyA=body1, bodyB=body2, distance=margin, physicsClientId=sim_id)
    else:
        results = pb.getClosestPoints(bodyA=body1, bodyB=body2, distance=margin)
    return len(results) != 0


def pb_set_joint_positions(body, joints: List[int], positions: List[float]):
    assert len(joints) == len(positions)
    for joint, position in zip(joints, positions):
        pb.resetJointState(body, joint, targetValue=position, targetVelocity=0)


def wiggle(source_obj: int, target_obj: int, max_tries: int=100000,
           sim_id: Optional[int]=None) -> Tuple[NPF64, NPF64]:
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


def trimesh_load_object(obj_path: str) -> trimesh.Trimesh:
    return trimesh.load(obj_path)


def trimesh_transform(mesh: trimesh.Trimesh, center: bool=True, 
                      scale: Optional[float]=None, rotation: Optional[NDArray]=None):
    
    # Automatically center. Also possibly rotate and scale.
    translation_matrix = np.eye(4)
    scaling_matrix = np.eye(4)
    rotation_matrix = np.eye(4)

    if center:
        t = mesh.centroid
        translation_matrix[:3, 3] = -t

    if scale is not None:
        scaling_matrix[0, 0] *= scale
        scaling_matrix[1, 1] *= scale
        scaling_matrix[2, 2] *= scale

    if rotation is not None:
        rotation_matrix[:3, :3] = rotation

    transform = np.matmul(scaling_matrix, np.matmul(rotation_matrix, translation_matrix))
    mesh.apply_transform(transform)


def trimesh_create_verts_volume(mesh: trimesh.Trimesh, voxel_size: float=0.015) -> NDArray[np.float32]:
    voxels = vcreate.voxelize(mesh, voxel_size)
    return np.array(voxels.points, dtype=np.float32)


def trimesh_create_verts_surface(mesh: trimesh.Trimesh, num_surface_samples: Optional[int]=1500) -> NDArray[np.float32]:
    surf_points, _ = trimesh.sample.sample_surface_even(
        mesh, num_surface_samples
    )
    return np.array(surf_points, dtype=np.float32)


def trimesh_get_vertices_and_faces(mesh: trimesh.Trimesh) -> Tuple[NDArray[np.float32], NDArray[np.int32]]:
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)
    return vertices, faces


def scale_points_circle(points: List[NDArray[np.float32]], base_scale: float=1.) -> List[NDArray[np.float32]]:
    points_cat = np.concatenate(points)
    assert len(points_cat.shape) == 2

    length = np.sqrt(np.sum(np.square(points_cat), axis=1))
    max_length = np.max(length, axis=0)

    new_points = []
    for p in points:
        new_points.append(base_scale * (p / max_length))

    return new_points


def cpd_transform(source, target, alpha: float=2.0) -> Tuple[NDArray, NDArray]:
    # reg = DeformableRegistration(**{ 'X': source, 'Y': target, 'tolerance':0.00001 })
    source, target = source.astype(np.float64), target.astype(np.float64)
    reg = deformable_registration(**{ 'X': source, 'Y': target, 'tolerance':0.00001 }, alpha=alpha)
    reg.register()
    #Returns the gaussian means and their weights - WG is the warp of source to target
    return reg.W, reg.G


def sst_cost_batch(source: NDArray, target: NDArray) -> float:

    idx = np.sum(np.abs(source[None, :] - target[:, None]), axis=2).argmin(axis=0)
    return np.mean(np.linalg.norm(source - target[idx], axis=1))  # TODO: test averaging instead of sum


def sst_cost_batch_pt(source, target):

    # for each vertex in source, find the closest vertex in target
    # we don't need to propagate the gradient here
    source_d, target_d = source.detach(), target.detach()
    indices = (source_d[:, :, None] - target_d[:, None, :]).square().sum(dim=3).argmin(dim=2)

    # go from [B, indices_in_target, 3] to [B, indices_in_source, 3] using target[batch_indices, indices]
    batch_indices = torch.arange(0, indices.size(0), device=indices.device)[:, None].repeat(1, indices.size(1))
    c = torch.sqrt(torch.sum(torch.square(source - target[batch_indices, indices]), dim=2))
    return torch.mean(c, dim=1)

    # simple version, about 2x slower
    # bigtensor = source[:, :, None] - target[:, None, :]
    # diff = torch.sqrt(torch.sum(torch.square(bigtensor), dim=3))
    # c = torch.min(diff, dim=2)[0]
    # return torch.mean(c, dim=1)


def warp_gen(canonical_index, objects, scale_factor=1., alpha: float=2.0, visualize=False):

    source = objects[canonical_index] * scale_factor
    targets = []
    for obj_idx, obj in enumerate(objects):
        if obj_idx != canonical_index:
            targets.append(obj * scale_factor)

    warps = []
    costs = []

    for target_idx, target in enumerate(targets):
        print("target {:d}".format(target_idx))

        w, g = cpd_transform(target, source, alpha=alpha)

        warp = np.dot(g, w)
        warp = np.hstack(warp)

        tmp = source + warp.reshape(-1, 3)
        costs.append(sst_cost_batch(tmp, target))

        warps.append(warp)

        if visualize:
            viz_utils.show_pcds_plotly({
                "target": target,
                "warp": source + warp.reshape(-1, 3),
            }, center=True)

    return warps, costs


def sst_pick_canonical(known_pts: List[NDArray[np.float32]]) -> int:

    # GPU acceleration makes this at least 100 times faster.
    known_pts = [torch.tensor(x, device="cuda:0", dtype=torch.float32) for x in known_pts]

    overall_costs = []
    for i in range(len(known_pts)):
        print(i)
        cost_per_target = []
        for j in range(len(known_pts)):
            if i != j:
                with torch.no_grad():
                    cost = sst_cost_batch_pt(known_pts[i][None], known_pts[j][None]).cpu()

                cost_per_target.append(cost.item())

        overall_costs.append(np.mean(cost_per_target))
    print("overall costs: {:s}".format(str(overall_costs)))
    return np.argmin(overall_costs)


def pca_transform(distances, n_dimensions=4):

    pca = PCA(n_components=n_dimensions)
    p_components = pca.fit_transform(np.array(distances))
    return p_components, pca
