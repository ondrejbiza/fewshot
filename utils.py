from typing import Tuple, Dict, List, Any, Optional
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation
import pybullet as pb
import open3d as o3d
import torch
from torch import nn
from torch import optim

from pybullet_planning.pybullet_tools import utils as pu
from exceptions import PlanningError, EnvironmentSetupError


class RealSenseD415():
  """Default configuration with 3 RealSense RGB-D cameras."""

  # Mimic RealSense D415 RGB-D camera parameters.
  image_size = (480, 640)
  intrinsics = (450., 0, 320., 0, 450., 240., 0, 0, 1)

  # Set default camera poses.
  front_position = (1., 0, 0.75)
  front_rotation = (np.pi / 4, np.pi, -np.pi / 2)
  front_rotation = pb.getQuaternionFromEuler(front_rotation)
  left_position = (0, 0.5, 0.75)
  left_rotation = (np.pi / 4.5, np.pi, np.pi / 4)
  left_rotation = pb.getQuaternionFromEuler(left_rotation)
  right_position = (0, -0.5, 0.75)
  right_rotation = (np.pi / 4.5, np.pi, 3 * np.pi / 4)
  right_rotation = pb.getQuaternionFromEuler(right_rotation)

  # Default camera configs.
  CONFIG = [{
      'image_size': image_size,
      'intrinsics': intrinsics,
      'position': front_position,
      'rotation': front_rotation,
      'zrange': (0.01, 10.),
      'noise': False
  }, {
      'image_size': image_size,
      'intrinsics': intrinsics,
      'position': left_position,
      'rotation': left_rotation,
      'zrange': (0.01, 10.),
      'noise': False
  }, {
      'image_size': image_size,
      'intrinsics': intrinsics,
      'position': right_position,
      'rotation': right_rotation,
      'zrange': (0.01, 10.),
      'noise': False
  }]


def render_camera(config):
    """Render RGB-D image with specified camera configuration."""

    # OpenGL camera settings.
    lookdir = np.float32([0, 0, 1]).reshape(3, 1)
    updir = np.float32([0, -1, 0]).reshape(3, 1)
    rotation = pb.getMatrixFromQuaternion(config['rotation'])
    rotm = np.float32(rotation).reshape(3, 3)
    lookdir = (rotm @ lookdir).reshape(-1)
    updir = (rotm @ updir).reshape(-1)
    lookat = config['position'] + lookdir
    focal_len = config['intrinsics'][0]
    znear, zfar = config['zrange']
    viewm = pb.computeViewMatrix(config['position'], lookat, updir)
    fovh = (config['image_size'][0] / 2) / focal_len
    fovh = 180 * np.arctan(fovh) * 2 / np.pi

    # Notes: 1) FOV is vertical FOV 2) aspect must be float
    aspect_ratio = config['image_size'][1] / config['image_size'][0]
    projm = pb.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

    # Render with OpenGL camera settings.
    _, _, color, depth, segm = pb.getCameraImage(
        width=config['image_size'][1],
        height=config['image_size'][0],
        viewMatrix=viewm,
        projectionMatrix=projm,
        shadow=1,
        flags=pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        # Note when use_egl is toggled, this option will not actually use openGL
        # but EGL instead.
        renderer=pb.ER_BULLET_HARDWARE_OPENGL)

    # Get color image.
    color_image_size = (config['image_size'][0], config['image_size'][1], 4)
    color = np.array(color, dtype=np.uint8).reshape(color_image_size)
    color = color[:, :, :3]  # remove alpha channel
    if config['noise']:
        color = np.int32(color)
        color += np.int32(np.random.normal(0, 3, config['image_size']))
        color = np.uint8(np.clip(color, 0, 255))

    # Get depth image.
    depth_image_size = (config['image_size'][0], config['image_size'][1])
    zbuffer = np.array(depth).reshape(depth_image_size)
    depth = (zfar + znear - (2. * zbuffer - 1.) * (zfar - znear))
    depth = (2. * znear * zfar) / depth
    if config['noise']:
        depth += np.random.normal(0, 0.003, depth_image_size)

    # Get segmentation image.
    segm = np.uint8(segm).reshape(depth_image_size)

    return color, depth, segm


def get_pointcloud(depth: NDArray, intrinsics: NDArray) -> NDArray:
  """Get 3D pointcloud from perspective depth image.

  Args:
    depth: HxW float array of perspective depth in meters.
    intrinsics: 3x3 float array of camera intrinsics matrix.

  Returns:
    points: HxWx3 float array of 3D points in camera coordinates.
  """
  height, width = depth.shape
  xlin = np.linspace(0, width - 1, width)
  ylin = np.linspace(0, height - 1, height)
  px, py = np.meshgrid(xlin, ylin)
  px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
  py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
  points = np.float32([px, py, depth]).transpose(1, 2, 0)
  return points


def transform_pointcloud(points, transform):
  """Apply rigid transformation to 3D pointcloud.

  Args:
    points: HxWx3 float array of 3D points in camera coordinates.
    transform: 4x4 float array representing a rigid transformation matrix.

  Returns:
    points: HxWx3 float array of transformed 3D points.
  """
  padding = ((0, 0), (0, 0), (0, 1))
  homogen_points = np.pad(points.copy(), padding,
                          'constant', constant_values=1)
  for i in range(3):
    points[Ellipsis, i] = np.sum(transform[i, :] * homogen_points, axis=-1)
  return points


def reconstruct_segmented_point_cloud(color, depth, segm, configs, object_ids) -> Tuple[Dict[int, NDArray], Dict[int, NDArray]]:

  object_points = {object_id: [] for object_id in object_ids}
  object_colors = {object_id: [] for object_id in object_ids}

  for tmp_color, tmp_depth, tmp_segm, tmp_config in zip(color, depth, segm, configs):

    intrinsics = np.array(tmp_config['intrinsics']).reshape(3, 3)
    xyz = get_pointcloud(tmp_depth, intrinsics)
    position = np.array(tmp_config['position']).reshape(3, 1)
    rotation = pb.getMatrixFromQuaternion(tmp_config['rotation'])
    rotation = np.array(rotation).reshape(3, 3)
    transform = np.eye(4)
    transform[:3, :] = np.hstack((rotation, position))
    xyz = transform_pointcloud(xyz, transform)

    xyz = xyz.reshape(-1, 3)
    tmp_segm = tmp_segm.reshape(-1)
    tmp_color = tmp_color.reshape(-1, 3)

    for object_id in object_ids:
      mask = tmp_segm == object_id
      if np.sum(mask) == 0:
        continue
      object_points[object_id].append(xyz[mask])
      object_colors[object_id].append(tmp_color[mask])

  for key in object_points.keys():
    object_points[key] = np.concatenate(object_points[key], axis=0)
    object_colors[key] = np.concatenate(object_colors[key], axis=0)

  return object_points, object_colors


def observe_point_cloud(
  camera_configs: List[Any], object_ids: List[int], n_points: Optional[int]=2000
) -> Tuple[Dict[int, NDArray], Dict[int, NDArray]]:

  c, d, s = [], [], []
  for cfg in camera_configs:
    out = render_camera(cfg)
    c.append(out[0])
    d.append(out[1])
    s.append(out[2])

  pcs, colors = reconstruct_segmented_point_cloud(c, d, s, camera_configs, object_ids)
  
  for key in pcs.keys():
    if n_points is not None and len(pcs[key]) > n_points:
      pcs[key], indices = farthest_point_sample(pcs[key], 2000)
      colors[key] = colors[key][indices]  # TODO: test

  return pcs, colors


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


def orthogonalize(x):
    """
    Produce an orthogonal frame from two vectors
    x: [B, 2, 3]
    """
    #u = torch.zeros([x.shape[0],3,3], dtype=torch.float32, device=x.device)
    u0 = x[:, 0] / torch.norm(x[:, 0], dim=1)[:, None]
    u1 = x[:, 1] - (torch.sum(u0 * x[:, 1], dim=1))[:, None] * u0
    u1 = u1 / torch.norm(u1, dim=1)[:, None]
    u2 = torch.cross(u0, u1, dim=1)
    return torch.stack([u0, u1, u2], dim=1)


def s2s2_pt(u, v):

    u = u / torch.norm(u, dim=0)
    v = v / torch.norm(v, dim=0)

    w1 = u
    w2 = v - torch.dot(u, v) * u
    w2 = w2 / torch.norm(w2)
    w3 = torch.cross(w1, w2)
    return torch.stack([w1, w2, w3], dim=0)


def s2s2_inverse(rot):

    return rot[0], rot[1]


def s2s2_batch_pt(u, v):

    u = u / torch.norm(u, dim=1)[:, None]
    v = v / torch.norm(v, dim=1)[:, None]

    w1 = u
    w2 = v - torch.bmm(u[:, None, :], v[:, :, None]) * u
    w2 = w2 / torch.norm(w2, dim=1)
    w3 = torch.cross(w1, w2, dim=1)
    return torch.stack([w1, w2, w3], dim=1)


def s2s2_batch_inverse(rot):

    return rot[:, 0], rot[:, 1]


def yaw_to_rot(yaw: NDArray) -> NDArray:

    c = np.cos(yaw)
    s = np.sin(yaw)

    t0 = np.zeros(1)
    t1 = np.ones(1)

    return np.stack([
        np.concatenate([c, -s, t0]),
        np.concatenate([s, c, t0]),
        np.concatenate([t0, t0, t1])
    ], axis=0)


def yaw_to_rot_pt(yaw: torch.Tensor) -> torch.Tensor:

    c = torch.cos(yaw)
    s = torch.sin(yaw)

    t0 = torch.zeros(1, device=c.device)
    t1 = torch.ones(1, device=c.device)

    return torch.stack([
        torch.cat([c, -s, t0]),
        torch.cat([s, c, t0]),
        torch.cat([t0, t0, t1])
    ], dim=0)


def create_o3d_pointcloud(points: NDArray, colors: Optional[NDArray]=None):

  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points)
  if colors is not None:
    pcd.colors = o3d.utility.Vector3dVector(colors)
  return pcd


def read_parameters(dbg_params):
    values = dict()
    for name, param in dbg_params.items():
        values[name] = pb.readUserDebugParameter(param)
    return values


def cost_pt(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

    diff = torch.sqrt(torch.sum(torch.square(source[:, None] - target[None, :]), dim=2))
    c = diff[list(range(len(diff))), torch.argmin(diff, dim=1)]
    return torch.mean(c)


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


def planar_pose_warp_gd(
  pca: PCA, canonical_obj: NDArray, points: NDArray, device: str="cuda:0", n_angles: int=10, 
  lr: float=1e-2, n_steps: int=100, object_size_reg: Optional[float]=None, verbose: bool=False
) -> Tuple[NDArray, float, Tuple[NDArray, NDArray, NDArray]]:
    # find planar pose and warping parameters of a canonical object to match a target point cloud
    start_angles = []
    for i in range(n_angles):
        angle = i * (2 * np.pi / n_angles)
        start_angles.append(angle)

    global_means = np.mean(points, axis=0)
    points = points - global_means[None]

    all_new_objects = []
    all_costs = []
    all_parameters = []
    device = torch.device(device)

    for trial_idx, start_pose in enumerate(start_angles):

        latent = nn.Parameter(torch.zeros(pca.n_components, dtype=torch.float32, device=device), requires_grad=True)
        center = nn.Parameter(torch.zeros(3, dtype=torch.float32, device=device), requires_grad=True)
        angle = nn.Parameter(
            torch.tensor([start_pose], dtype=torch.float32, device=device),
            requires_grad=True
        )
        means = torch.tensor(pca.mean_, dtype=torch.float32, device=device)
        components = torch.tensor(pca.components_, dtype=torch.float32, device=device)
        canonical_obj_pt = torch.tensor(canonical_obj, dtype=torch.float32, device=device)
        points_pt = torch.tensor(points, dtype=torch.float32, device=device)

        opt = optim.Adam([latent, center, angle], lr=lr)

        for i in range(n_steps):

            opt.zero_grad()

            rot = yaw_to_rot_pt(angle)

            deltas = (torch.matmul(latent[None, :], components)[0] + means).reshape((-1, 3))
            new_obj = canonical_obj_pt + deltas
            # cost = utils.cost_pt(torch.matmul(rot, (points_pt - center[None]).T).T, new_obj)
            cost = cost_pt(points_pt, torch.matmul(rot, new_obj.T).T + center[None])

            if object_size_reg is not None:
              size = torch.max(torch.sqrt(torch.sum(torch.square(new_obj), dim=1)))
              cost = cost + object_size_reg * size

            if verbose:
                print("cost:", cost)

            cost.backward()
            opt.step()

        with torch.no_grad():

            deltas = (torch.matmul(latent[None, :], components)[0] + means).reshape((-1, 3))
            rot = yaw_to_rot_pt(angle)
            new_obj = canonical_obj_pt + deltas
            new_obj = torch.matmul(rot, new_obj.T).T
            new_obj = new_obj + center[None]
            new_obj = new_obj.cpu().numpy()

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(np.concatenate([points_pt.cpu().numpy(), new_obj], axis=0))
            # pcd.colors = o3d.utility.Vector3dVector(np.concatenate([np.zeros_like(points_pt.cpu().numpy()) + 0.9, np.zeros_like(new_obj)], axis=0))
            # utils.o3d_visualize(pcd)

            new_obj = new_obj + global_means[None]

            all_costs.append(cost.item())
            all_new_objects.append(new_obj)
            all_parameters.append((latent.cpu().numpy(), center.cpu().numpy() + global_means, angle.cpu().numpy()))

    best_idx = np.argmin(all_costs)
    return all_new_objects[best_idx], all_costs[best_idx], all_parameters[best_idx]


def planar_pose_gd(canonical_obj: NDArray, points: NDArray, device: str="cuda:0", n_angles: int=10, 
                   lr: float=1e-2, n_steps: int=100, verbose: bool=False) -> Tuple[NDArray, float, Tuple[NDArray, NDArray]]:
    # find planar pose and warping parameters of a canonical object to match a target point cloud
    assert n_angles > 0 and n_steps > 0

    start_angles = []
    for i in range(n_angles):
        angle = i * (2 * np.pi / n_angles)
        start_angles.append(angle)

    global_means = np.mean(points, axis=0)
    points = points - global_means[None]

    all_new_objects = []
    all_costs = []
    all_parameters = []
    device = torch.device(device)

    for trial_idx, start_pose in enumerate(start_angles):

        center = nn.Parameter(torch.zeros(3, dtype=torch.float32, device=device), requires_grad=True)
        angle = nn.Parameter(
            torch.tensor([start_pose], dtype=torch.float32, device=device),
            requires_grad=True
        )
        canonical_obj_pt = torch.tensor(canonical_obj, dtype=torch.float32, device=device)
        points_pt = torch.tensor(points, dtype=torch.float32, device=device)

        opt = optim.Adam([center, angle], lr=lr)

        for i in range(n_steps):

            opt.zero_grad()

            rot = yaw_to_rot_pt(angle)

            cost = cost_pt(points_pt, torch.matmul(rot, canonical_obj_pt.T).T + center[None])

            if verbose:
                print("cost:", cost)

            cost.backward()
            opt.step()

        with torch.no_grad():

            rot = yaw_to_rot_pt(angle)
            new_obj = torch.matmul(rot, canonical_obj_pt.T).T
            new_obj = new_obj + center[None]
            new_obj = new_obj.cpu().numpy()

            new_obj = new_obj + global_means[None]

            all_costs.append(cost.item())
            all_new_objects.append(new_obj)
            all_parameters.append((center.cpu().numpy(), angle.cpu().numpy()))

    best_idx = np.argmin(all_costs)
    return all_new_objects[best_idx], all_costs[best_idx], all_parameters[best_idx]


def move_hand_back(pose: Tuple[NDArray, NDArray], delta: float) -> Tuple[NDArray, NDArray]:

    pos, quat = pose
    rot = pu.matrix_from_quat(quat)
    vec = np.array([0., 0., delta], dtype=np.float32)
    vec = np.matmul(rot, vec)
    pos = pos - vec
    return pos, quat


def wiggle(source_obj: int, target_obj: int, max_tries: int=100000) -> Tuple[NDArray, NDArray]:
  """Wiggle the source object out of a collision with the target object.
  
  Important: this function will change the state of the world and we assume
  the world was saved before and will be restored after.
  """
  i = 0
  pos, quat = pu.get_pose(source_obj)
  
  pb.performCollisionDetection()
  in_collision = pu.body_collision(source_obj, target_obj)
  if not in_collision:
    return pos, quat
  
  while True:

    new_pos = pos + np.random.normal(0, 0.01, 3)
    pu.set_pose(source_obj, (new_pos, quat))

    pb.performCollisionDetection()
    in_collision = pu.body_collision(source_obj, target_obj)
    if not in_collision:
      break

    i += 1
    if i > max_tries:
      raise PlanningError("Could not wiggle object out of collision.")

  return new_pos, quat


def place_object(object: int, floor: int, placed_objects: List[int], 
                 workspace_low: NDArray, workspace_high: NDArray,
                 random_rotations: bool=True, min_distance_between_objects: float=0.2,
                 max_tries: int=10001):

    n_tries = 0

    while True:

        n_tries += 1
        if n_tries == max_tries:
            raise EnvironmentSetupError("Cannot place an object.")

        yaw = 0.
        if random_rotations:
          yaw = np.random.uniform(0, 2 * np.pi)
        quat = pu.quat_from_euler(pu.Euler(yaw=yaw))

        # set random position
        x = np.random.uniform(workspace_low[0], workspace_high[0])
        y = np.random.uniform(workspace_low[1], workspace_high[1])
        pu.set_pose(object, (pu.Point(x=x, y=y, z=pu.stable_z(object, floor)), quat))

        # get bbox
        bbox_min, bbox_max = pb.getAABB(object)

        # check if objects overlap
        overlapping = pb.getOverlappingObjects(bbox_min, bbox_max)
        flag = False
        for pair in overlapping:
            tmp = pair[0]
            if tmp != object and tmp != floor:
                flag = True
                break
        if flag:
            # print("Objects are overlapping.")
            continue

        # check for minimum distance between objects
        flag = False
        for tmp in placed_objects:
            tmp_x, tmp_y = pu.get_pose(tmp)[0][:2]
            l2 = np.sqrt((x - tmp_x)**2 + (y - tmp_y)**2)
            if l2 < min_distance_between_objects:
                flag = True
        if flag:
            # print("Objects too close.")
            continue

        # check if the object is out of bounds
        if (bbox_min[0] < workspace_low[0] or bbox_min[1] < workspace_low[1] or
            bbox_max[0] > workspace_high[0] or bbox_max[1] > workspace_high[1]):
            # print("Object bbox outside of bounds.")
            continue

        break


def transform_pointcloud_2(cloud: NDArray, T: NDArray, is_position: bool=True) -> NDArray:
  # TODO: what's up with transform_pointcloud?
  n = cloud.shape[0]
  cloud = cloud.T
  augment = np.ones((1, n)) if is_position else np.zeros((1, n))
  cloud = np.concatenate((cloud, augment), axis=0)
  cloud = np.dot(T, cloud)
  # TODO: divide by the fourth coordinate?
  cloud = cloud[0: 3, :].T
  return cloud


def rotate_for_open3d(pc):
    # Going from realsense to open3d, the point cloud will be upside down and rotated opposite to the camera angle.
    mat = Rotation.from_euler("zyx", (np.pi, np.pi, 0.)).as_matrix()
    return np.matmul(pc, mat)


def update_open3d_pointcloud(pcd: o3d.geometry.PointCloud, vertices: np.ndarray, colors: Optional[np.ndarray]=None):

    pcd.points = o3d.utility.Vector3dVector(vertices)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
