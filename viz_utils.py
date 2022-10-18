import numpy as np
import open3d as o3d


def show_pointcloud(points, colors=None):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(list(points.values()), axis=0).reshape((-1, 3)))
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.concatenate(list(colors.values()), axis=0).reshape((-1, 3)) / 255.)
    
    o3d.visualization.draw([pcd])