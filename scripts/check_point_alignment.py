
import torch
import argparse
import numpy as np
import os
from data_utils.ShapeNetDataLoader import PartNormalDataset
from src import utils, viz_utils
from scipy.spatial.transform import Rotation

#load the pointnet one
num_points = 2048

root = 'Pointnet_Pointnet2_pytorch/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'
mug_id = '03797390' 
file_folder  = 'mug_centered_obj_normalized'
first_mug_name = '1a1c0a8d4bad82169f0594e65f756cf5'
first_mug_path = os.path.join(root, mug_id, first_mug_name + '.txt')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

#def load_segmented_pcl():

def load_pcl_text_file(fn):
    cls = 'Mug'
    data = np.loadtxt(fn).astype(np.float32)
    #if not self.normal_channel:
    point_set = data[:, 0:3]
    #else:
        #point_set = data[:, 0:6]
    seg = data[:, -1].astype(np.int32)

    #point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

    choice = np.random.choice(len(seg), num_points, replace=True)
    # resample
    #point_set = point_set[choice, :]
    #seg = seg[choice]

    return point_set, cls, seg


seg_points, _, seg = load_pcl_text_file(first_mug_path)

# TEST_DATASET = PartNormalDataset(root=root, npoints=2048, split='test', normal_channel=True)
# testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False,)

file_name = '/Users/skye/relational_ndf/src/rndf_robot/descriptions/objects/mug_centered_obj_normalized/1a1c0a8d4bad82169f0594e65f756cf5/models/model_normalized.obj'
mesh = utils.trimesh_load_object(file_name)
non_seg_points = mesh.vertices

print(seg_points.shape)
rotation = Rotation.from_euler("zyx", [0., np.pi / 2., 0]).as_quat()

transform = utils.pos_quat_to_transform([0,0,0], rotation)


seg_points = utils.transform_pcd(seg_points, transform)
def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    return pc

#seg_points = utils.scale_points_circle([seg_points], base_scale=0.44)[0]
seg_points = pc_normalize(seg_points)

seg_handle = seg_points[seg==36]
seg_cup = seg_points[seg==37]

viz_utils.show_pcds_pyplot({'seg_handle':seg_handle, 'seg_cup':seg_cup, 'non_seg':non_seg_points})




# mug_id = '03797390' 
# file_folder  = 'mug_centered_obj_normalized'
# first_mug_name = '1a1c0a8d4bad82169f0594e65f756cf5'

#load the relndf one

#Print number of points
#Plot them both
