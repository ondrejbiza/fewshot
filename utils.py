import numpy as np
import pybullet as pb
import open3d as o3d
import torch


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


def get_pointcloud(depth, intrinsics):
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


def reconstruct_segmented_point_cloud(color, depth, segm, configs, object_ids):

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


def farthest_point_sample(point, npoint):
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


def yaw_to_rot_pt(yaw):

    c = torch.cos(yaw)
    s = torch.sin(yaw)

    t0 = torch.zeros(1, device=c.device)
    t1 = torch.ones(1, device=c.device)

    return torch.stack([
        torch.cat([c, -s, t0]),
        torch.cat([s, c, t0]),
        torch.cat([t0, t0, t1])
    ], dim=0)


def o3d_visualize(pcd):

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


def cost_pt(source, target):

    diff = torch.sqrt(torch.sum(torch.square(source[:, None] - target[None, :]), dim=2))
    c = diff[list(range(len(diff))), torch.argmin(diff, dim=1)]
    return torch.mean(c)