
#get the original object
from src import utils, viz_utils
import os
import numpy as np
from scipy.spatial.transform import Rotation

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
    point_set = utils.scale_points_circle([point_set], base_scale=0.1)[0]
    #choice = np.random.choice(len(seg_ids), num_points, replace=True)
    return point_set, cls, seg_ids

def load_unsegmented_pointcloud(obj_file_path): 
    num_surface_samples = 10000
    mesh = utils.trimesh_load_object(obj_file_path)
    center_translation = mesh.centroid
    #centers.append(utils.pos_quat_to_transform(center_translation, (0,0,0,1)))
    utils.trimesh_transform(mesh, center=True, scale=None, rotation=None)
    #meshes.append(mesh)
    sp = utils.trimesh_create_verts_surface(mesh, num_surface_samples=num_surface_samples)
    ssp = utils.trimesh_create_verts_surface(mesh, num_surface_samples=2000)
    mp, f = utils.trimesh_get_vertices_and_faces(mesh)
    ssp, sp, mp = utils.scale_points_circle([ssp, sp, mp], base_scale=0.1)
    h = np.concatenate([mp, sp])  # Order important!
    return h, mp,  mesh

obj_id = '5c7c4cb503a757147dbda56eabff0c47'
obj_file_path = f"/Users/skye/relational_ndf/src/rndf_robot/descriptions/objects/mug_centered_obj_normalized/{obj_id}/models/model_normalized.obj"

unseg_h, unseg_mp, unseg_mesh = load_unsegmented_pointcloud(obj_file_path)
seg_pcl, _, seg_ids = load_segmented_pointcloud_from_txt(obj_id)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import neighbors 

# we create 40 separable points
np.random.seed(0)
X = seg_pcl
Y = seg_ids - 36

print(np.max(Y))
print(np.min(Y))


# fit the model
clf = neighbors.KNeighborsClassifier(2)#svm.SVC()
clf.fit(X, Y)


unseg_ids = clf.predict(unseg_h)
print(np.max(unseg_ids))
print(np.min(unseg_ids))
# print(unseg_ids)
# print(unseg_h.shape)
# print(unseg_ids.shape)
viz_utils.show_pcds_plotly({'unseg_cup': unseg_h[unseg_ids==1], 'unseg_handle': unseg_h[unseg_ids==0],
                            'seg_cup': seg_pcl[Y==1], 'seg_handle': seg_pcl[Y==0]})

mask = unseg_ids==0

face_mask = mask[unseg_mesh.faces].all(axis=1)
unseg_mesh.update_faces(face_mask)
unseg_mesh.show()


