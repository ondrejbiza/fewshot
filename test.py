import numpy as np
from src.utils import farthest_point_sample


delta = 0.01
source_pcd = np.random.normal(0, 1, (12481, 3))
target_pcd = np.random.normal(0, 1, (9832, 3))

dist = np.sqrt(np.sum(np.square(source_pcd[:, None] - target_pcd[None]), axis=-1))
indices = np.where(dist <= delta)
source_points = source_pcd[indices[0]]
target_points = target_pcd[indices[1]]

source_points, indices2 = farthest_point_sample(source_points, 5)
target_points = target_points[indices2]

print(source_points.shape, target_points.shape)
