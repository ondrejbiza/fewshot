import matplotlib
import torch

from src.object_warping import ObjectWarpingSE2Batch, ObjectSE2Batch, ObjectWarpingSE3Batch, warp_to_pcd_se2, warp_to_pcd_se3
from src import utils, viz_utils


class NDFInterface:

    def __init__(self):

        pass

    def set_demo_info(self, pc_master_dict, cfg, n_demos):
        matplotlib.use("WebAgg")
        # Get a single demonstration.
        mug_pcd = pc_master_dict["child"]["demo_start_pcds"][0]
        if len(mug_pcd) > 2000:
            mug_pcd, _ = utils.farthest_point_sample(mug_pcd, 2000)
        canon_mug = utils.CanonObj.from_pickle("data/ndf_mugs_pca_8_dim.npy")

        # Perception.
        warp = ObjectWarpingSE2Batch(
            canon_mug, mug_pcd, torch.device("cuda:0"), lr=1e-2, n_steps=100,
            n_samples=None, object_size_reg=0.0)
        mug_pcd_complete, _, mug_param = warp_to_pcd_se2(warp, n_angles=12, n_batches=1)

        viz_utils.show_pcds_pyplot({
            "pcd": mug_pcd,
            "warp": mug_pcd_complete
        }, center=False)

        # Save the mesh and its convex decomposition.
        mesh = canon_mug.to_mesh(mug_param)
        mesh.export("tmp.stl")
        utils.convex_decomposition(mesh, "tmp.obj")

    def optimize_transform_implicit(self, target_obj_pcd_obs, ee: bool):

        pass
