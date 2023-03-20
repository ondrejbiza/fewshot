import os

import numpy as np  
from scipy.spatial.transform import Rotation
import torch

from src.object_warping import ObjectWarpingSE2Batch, ObjectWarpingSE3Batch, warp_to_pcd_se2, warp_to_pcd_se3, PARAM_1
from src import utils, viz_utils


obj_path = "data/ndf_objects/mug_centered_obj_normalized/9c930a8a3411f069e7f67f334aa9295c/models/model_normalized.obj"
rotation = Rotation.from_euler("zyx", [0., 0., np.pi / 2]).as_matrix()

mesh = utils.trimesh_load_object(obj_path)
utils.trimesh_transform(mesh, center=True, scale=None, rotation=rotation)

sp = utils.trimesh_create_verts_surface(mesh, num_surface_samples=2000)
sp = utils.scale_points_circle([sp], base_scale=0.1)[0]

sp = sp[sp[:, 1] >= 0]

canon_source = utils.CanonObj.from_pickle("data/230315_ndf_mugs_scale_pca_8_dim_alp_0_01.pkl")
warp = ObjectWarpingSE2Batch(
    canon_source, sp, torch.device("cuda:0"), **PARAM_1,
    init_scale=0.7)
source_pcd_complete, _, source_param = warp_to_pcd_se2(warp, n_angles=12, n_batches=1)

viz_utils.show_pcds_plotly({
    "0": sp,
    "1": source_pcd_complete
})

dir_name = "warping_figure_5"
if not os.path.isdir(dir_name):
    os.makedirs(dir_name)

obj_param = utils.ObjParam(latent=np.zeros(canon_source.n_components, dtype=np.float32))
canon_pcd = canon_source.to_pcd(obj_param)

viz_utils.save_o3d_pcd(canon_pcd, os.path.join(dir_name, "canonical.pcd"))
viz_utils.save_o3d_pcd(source_pcd_complete, os.path.join(dir_name, "warped.pcd"))
viz_utils.save_o3d_pcd(sp, os.path.join(dir_name, "partial.pcd"))
