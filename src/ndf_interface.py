import matplotlib
import pybullet as pb
import torch

from src.object_warping import ObjectWarpingSE2Batch, ObjectSE2Batch, ObjectWarpingSE3Batch, ObjectSE3Batch, warp_to_pcd_se2, warp_to_pcd_se3
from src import utils, viz_utils
from pybullet_planning.pybullet_tools import utils as pu


class NDFInterface:

    def __init__(self):

        pass

    def set_demo_info(self, pc_master_dict, cfg, n_demos):
        matplotlib.use("WebAgg")
        print(pc_master_dict["child"].keys())
        # Get a single demonstration.
        mug_pcd = pc_master_dict["child"]["demo_final_pcds"][0]
        tree_pcd = pc_master_dict["parent"]["demo_start_pcds"][0]
        if len(mug_pcd) > 2000:
            mug_pcd, _ = utils.farthest_point_sample(mug_pcd, 2000)
        if len(tree_pcd) > 2000:
            tree_pcd, _ = utils.farthest_point_sample(tree_pcd, 2000)
        canon_mug = utils.CanonObj.from_pickle("data/230201_ndf_mugs_large_pca_8_dim.npy")
        canon_tree =utils.CanonObj.from_pickle("data/real_tree_pc.pkl")

        # Perception.
        warp = ObjectWarpingSE3Batch(
            canon_mug, mug_pcd, torch.device("cuda:0"), lr=1e-2, n_steps=100,
            n_samples=1000, object_size_reg=0.1, scaling=True)
        mug_pcd_complete, _, mug_param = warp_to_pcd_se3(warp, n_angles=30, n_batches=5)
        # See how tiny the canonical mug is:
        # warp = ObjectSE3Batch(
        #     canon_mug, mug_pcd, torch.device("cuda:0"), lr=1e-2, n_steps=100,
        #     n_samples=1000)
        # mug_pcd_complete, _, mug_param = warp_to_pcd_se3(warp, n_angles=30, n_batches=5)

        viz_utils.show_pcds_pyplot({
            "pcd": mug_pcd,
            "warp": mug_pcd_complete
        }, center=False)

        warp = ObjectSE2Batch(
            canon_tree, tree_pcd, torch.device("cuda:0"), lr=1e-2, n_steps=100,
            n_samples=None)
        tree_pcd_complete, _, tree_param = warp_to_pcd_se2(warp, n_angles=12, n_batches=1)

        viz_utils.show_pcds_pyplot({
            "pcd": tree_pcd,
            "warp": tree_pcd_complete
        }, center=False)

        # Save the mesh and its convex decomposition.
        mesh = canon_mug.to_mesh(mug_param)
        mesh.export("tmp_source.stl")
        utils.convex_decomposition(mesh, "tmp_source.obj")

        mesh = canon_tree.to_mesh(tree_param)
        mesh.export("tmp_target.stl")
        utils.convex_decomposition(mesh, "tmp_target.obj")

        # Add predicted meshes to pybullet.
        source_pb = pb.loadURDF("tmp_source.urdf", useFixedBase=True)
        pu.set_pose(source_pb, (mug_param.position, mug_param.quat))

        target_pb = pb.loadURDF("tmp_target.urdf", useFixedBase=True)
        pu.set_pose(target_pb, (tree_param.position, tree_param.quat))

        while True:
            pass

    def optimize_transform_implicit(self, target_obj_pcd_obs, ee: bool):

        pass
