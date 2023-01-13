from typing import Optional, Dict
import numpy as np
from numpy.typing import NDArray
import open3d as o3d


def o3d_visualize(pcd):

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


def show_pointcloud(points: NDArray, colors: Optional[NDArray]=None):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(list(points.values()), axis=0).reshape((-1, 3)))
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.concatenate(list(colors.values()), axis=0).reshape((-1, 3)) / 255.)
    
    o3d.visualization.draw([pcd])


def show_scene(point_clouds: Dict[int, NDArray], background: Optional[NDArray]=None):
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
        background_colors[:] = 0.7
        point_colors = np.concatenate([point_colors, background_colors], axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    o3d_visualize(pcd)


def draw_square(img: np.ndarray, x: int, y: int, square_size=20, copy=False, intensity: float=1.):

    size = square_size // 2
    x_limits = [x - size, x + size]
    y_limits = [y - size, y + size]
    for i in range(len(x_limits)):
        x_limits[i] = min(img.shape[0], max(0, x_limits[i]))
    for i in range(len(y_limits)):
        y_limits[i] = min(img.shape[1], max(0, y_limits[i]))

    if copy:
        img = np.array(img, dtype=img.dtype)

    if img.dtype == np.uint8:
        img[x_limits[0]: x_limits[1], y_limits[0]: y_limits[1]] = int(255 * intensity)
    else:
        img[x_limits[0]: x_limits[1], y_limits[0]: y_limits[1]] = intensity

    return img
