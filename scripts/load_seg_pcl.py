
from rndf_robot.share.globals import bad_shapenet_mug_ids_list
from rndf_robot.utils import util, path_util
import os, os.path as osp
import numpy as np
from scipy.spatial.transform import Rotation
from src import utils
import pickle

def center_pcl(pcl, return_centroid=False):
    l = pcl.shape[0]
    centroid = np.mean(pcl, axis=0)
    pcl = pcl - centroid
    if return_centroid:
        return pcl, centroid
    else:
        return pcl

def load_segmented_pointcloud_from_txt(pcl_id, num_points=2048, root = 'Pointnet_Pointnet2_pytorch/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/03797390'):
    fn = os.path.join(root, pcl_id + '.txt')
    cls = 'Mug'
    data = np.loadtxt(fn).astype(np.float32)
    #if not self.normal_channel:
    point_set = data[:, 0:3]
    #else:
        #point_set = data[:, 0:6]
    seg_ids = data[:, -1].astype(np.int32)

    point_set[:, 0:3] = center_pcl(point_set[:, 0:3])

    rotation = Rotation.from_euler("zyx", [0., np.pi/2, 0.]).as_quat()

    transform = utils.pos_quat_to_transform([0,0,0], rotation)
    point_set = utils.transform_pcd(point_set, transform)

    choice = np.random.choice(len(seg_ids), num_points, replace=True)
    # resample
    #point_set = point_set[choice, :]
    #seg = seg[choice]

    return point_set, cls, seg_ids



#get the list of ids
#get rids of the bad ids
#load all the mugs

bad_ids = {
    'mug': bad_shapenet_mug_ids_list,
}

mesh_data_dirs = {
    'mug': 'mug_centered_obj_normalized',
}
mesh_data_dirs = {k: osp.join(path_util.get_rndf_obj_descriptions(), v) for k, v in mesh_data_dirs.items()}

for k, v in mesh_data_dirs.items():
    # get train samples
    objects_raw = os.listdir(v) 
    objects_filtered = [fn for fn in objects_raw if (fn.split('/')[-1] not in bad_ids[k] and '_dec' not in fn)]

all_pointclouds = {}
for obj in objects_filtered:
	try: 
		pointcloud, cls, seg = load_segmented_pointcloud_from_txt(obj)
	except FileNotFoundError as e:
		continue
	seg -= 36
	colors = np.zeros((seg.shape[0], 3))
	colors[:, 0] = np.ones(seg.shape[0])
	colored_labeled_pointcloud = np.concatenate([pointcloud, colors, np.atleast_2d(seg).T], axis=-1)
	all_pointclouds[obj] = colored_labeled_pointcloud

pickle.dump(all_pointclouds, open('segmented_mugs.pkl', 'wb'))
def apply_random_SE2_rotation(pcl):
	pass


if __name__=="__main__":
	pass