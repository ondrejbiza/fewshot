
import os, os.path as osp
from src import utils, viz_utils
from rndf_robot.utils import util, path_util
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import pickle

#For loading the part-segmented shapenet mugs
#That's currently hardcoded into the path
def load_segmented_pointcloud_from_txt(pcl_id, num_points=2048, 
                                       root = 'Pointnet_Pointnet2_pytorch/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/03797390'):
    fn = os.path.join(root, pcl_id + '.txt')
    cls = 'Mug'
    data = np.loadtxt(fn).astype(np.float32)
    #if not self.normal_channel: #<-- ignore the normals
    point_set = data[:, 0:3]
    #else:
        #point_set = data[:, 0:6]
    seg_ids = data[:, -1].astype(np.int32)
    point_set[:, 0:3] = utils.center_pcl(point_set[:, 0:3])

    #fixed transform to align with the other mugs being used
    rotation = Rotation.from_euler("zyx", [0., np.pi/2, 0.]).as_quat()
    transform = utils.pos_quat_to_transform([0,0,0], rotation)
    point_set = utils.transform_pcd(point_set, transform)

    choice = np.random.choice(len(seg_ids), num_points, replace=True)
    return point_set, cls, seg_ids

mesh_data_dirs = {
    'mug': 'mug_centered_obj_normalized',
    'bottle': 'bottle_centered_obj_normalized',
    'bowl': 'bowl_centered_obj_normalized',
    'syn_rack_easy': 'syn_racks_easy_obj',
    'syn_container': 'box_containers_unnormalized'
}
mesh_data_dirs = {k: osp.join(path_util.get_rndf_obj_descriptions(), v) for k, v in mesh_data_dirs.items()}


obj_id = 'e9bd4ee553eb35c1d5ccc40b510e4bd'
obj_class = 'mug'
#For shapenet objects
pcl_root = 'Pointnet_Pointnet2_pytorch/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/03797390'
mesh_root = mesh_data_dirs[obj_class]
seg_root = 'Pointnet_Pointnet2_pytorch/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/03797390'

#For simulated objects 
pcl_root = 'sim_objects/mugs/a4b3pawyixj5szpb_mug_points.pkl'
mesh_root = 'sim_objects/mugs/a4b3pawyixj5szpb_mesh.obj'
seg_root = 'sim_objects/mugs/a4b3pawyixj5szpb_seg_ids.pkl'

#load pcl
pcl, _, seg_ids = load_segmented_pointcloud_from_txt(obj_id)

sim_pcl = pickle.load(open(pcl_root, 'rb'))
sim_seg = pickle.load(open(seg_root, 'rb'))


# #load mesh
# obj_file = osp.join(mesh_data_dirs[obj_class], obj_id, 'models/model_normalized.obj')
# obj_file_dec = obj_file.split('.obj')[0] + '_dec.obj'
# mesh = trimesh.load(obj_file)
# mesh_decomposition = trimesh.load(obj_file_dec)

#load segmentation
viz_utils.show_pcds_plotly({'Full PCL': pcl, 'Full sim PCL': sim_pcl})
viz_utils.show_pcds_plotly({f'{i}': pcl[seg_ids==i] for i in np.unique(seg_ids)} + {f'sim_{i}': sim_pcl[sim_seg==i] for i in np.unique(sim_seg)})
#viz_utils.show_meshes_plotly({"Full Mesh": mesh.vertices}, {"Full Mesh": mesh.faces})
#viz_utils.show_meshes_plotly({"Decomposition": mesh_decomposition.vertices},{"Decomposition": mesh_decomposition.faces})

