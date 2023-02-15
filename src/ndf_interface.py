import numpy as np
import matplotlib
import pybullet as pb
import torch

from src.object_warping import ObjectWarpingSE2Batch, ObjectSE2Batch, ObjectWarpingSE3Batch, ObjectSE3Batch, warp_to_pcd_se2, warp_to_pcd_se3
from src import demo, utils, viz_utils
from pybullet_planning.pybullet_tools import utils as pu


class NDFInterface:

    def __init__(self):

        self.canon_mug = utils.CanonObj.from_pickle("data/230213_ndf_mugs_scale_large_pca_8_dim_alp_0.01.pkl")
        self.canon_tree = utils.CanonObj.from_pickle("data/230213_ndf_trees_scale_large_pca_8_dim_alp_2.pkl")

    def set_demo_info(self, pc_master_dict, cfg, n_demos, show: bool=False):
        demo_idx = 0

        # Get a single demonstration.
        mug_pcd = pc_master_dict["child"]["demo_start_pcds"][demo_idx]
        mug_start = np.array(pc_master_dict["child"]["demo_start_poses"][demo_idx], dtype=np.float64)
        mug_final = np.array(pc_master_dict["child"]["demo_final_poses"][demo_idx], dtype=np.float64)

        mug_start_pos, mug_start_quat = mug_start[:3], mug_start[3:]
        mug_final_pos, mug_final_quat = mug_final[:3], mug_final[3:]
        mug_start_trans = utils.pos_quat_to_transform(mug_start_pos, mug_start_quat)
        mug_final_trans = utils.pos_quat_to_transform(mug_final_pos, mug_final_quat)
        mug_start_to_final = mug_final_trans @ np.linalg.inv(mug_start_trans)

        tree_pcd = pc_master_dict["parent"]["demo_start_pcds"][demo_idx]
        if len(mug_pcd) > 2000:
            mug_pcd, _ = utils.farthest_point_sample(mug_pcd, 2000)
        if len(tree_pcd) > 2000:
            tree_pcd, _ = utils.farthest_point_sample(tree_pcd, 2000)

        # Perception.
        warp = ObjectWarpingSE2Batch(
            self.canon_mug, mug_pcd, torch.device("cuda:0"), lr=1e-2, n_steps=100,
            n_samples=1000, object_size_reg=0.1, scaling=True, init_scale=0.1)
        mug_pcd_complete, _, mug_param = warp_to_pcd_se2(warp, n_angles=12, n_batches=1)

        warp = ObjectWarpingSE2Batch(
            self.canon_tree, tree_pcd, torch.device("cuda:0"), lr=1e-2, n_steps=100,
            n_samples=1000, object_size_reg=0.1, scaling=True, init_scale=0.1)
        tree_pcd_complete, _, tree_param = warp_to_pcd_se2(warp, n_angles=12, n_batches=1)

        if show:
            viz_utils.show_pcds_plotly({
                "pcd": mug_pcd,
                "warp": mug_pcd_complete
            }, center=False)
            viz_utils.show_pcds_plotly({
                "pcd": tree_pcd,
                "warp": tree_pcd_complete
            }, center=False)

        # Move object to final pose.
        trans = utils.pos_quat_to_transform(mug_param.position, mug_param.quat)
        trans = mug_start_to_final @ trans
        pos, quat = utils.transform_to_pos_quat(trans)
        mug_param.position = pos
        mug_param.quat = quat

        # Save the mesh and its convex decomposition.
        mesh = self.canon_mug.to_mesh(mug_param)
        mesh.export("tmp_source.stl")
        utils.convex_decomposition(mesh, "tmp_source.obj")

        mesh = self.canon_tree.to_mesh(tree_param)
        mesh.export("tmp_target.stl")
        utils.convex_decomposition(mesh, "tmp_target.obj")

        # Add predicted meshes to pybullet.
        source_pb = pb.loadURDF("tmp_source.urdf", useFixedBase=True)
        pu.set_pose(source_pb, (mug_param.position, mug_param.quat))

        target_pb = pb.loadURDF("tmp_target.urdf", useFixedBase=True)
        pu.set_pose(target_pb, (tree_param.position, tree_param.quat))

        self.knns, self.deltas, self.target_indices = demo.save_place_nearby_points(
            source_pb, target_pb, self.canon_mug, mug_param, self.canon_tree, tree_param, 0.1)

        pb.removeBody(source_pb)
        pb.removeBody(target_pb)

    def infer_relpose(self, source_pcd, target_pcd, se3: bool=False, show: bool=False):

        if len(source_pcd) > 2000:
            source_pcd, _ = utils.farthest_point_sample(source_pcd, 2000)
        if len(target_pcd) > 2000:
            target_pcd, _ = utils.farthest_point_sample(target_pcd, 2000)

        if se3:
            warp = ObjectWarpingSE3Batch(
                self.canon_mug, source_pcd, torch.device("cuda:0"), lr=1e-2, n_steps=100,
                n_samples=1000, object_size_reg=0.1, scaling=True, init_scale=0.1)
            mug_pcd_complete, _, mug_param = warp_to_pcd_se3(warp, n_angles=12, n_batches=15)
        else:
            warp = ObjectWarpingSE2Batch(
                self.canon_mug, source_pcd, torch.device("cuda:0"), lr=1e-2, n_steps=100,
                n_samples=1000, object_size_reg=0.1, scaling=True, init_scale=0.1)
            mug_pcd_complete, _, mug_param = warp_to_pcd_se2(warp, n_angles=12, n_batches=1)

        warp = ObjectWarpingSE2Batch(
            self.canon_tree, target_pcd, torch.device("cuda:0"), lr=1e-2, n_steps=100,
            n_samples=1000, object_size_reg=0.1, scaling=True, init_scale=0.1)
        tree_pcd_complete, _, tree_param = warp_to_pcd_se2(warp, n_angles=12, n_batches=1)

        if show:
            viz_utils.show_pcds_plotly({
                "pcd": source_pcd,
                "warp": mug_pcd_complete
            }, center=False)
            viz_utils.show_pcds_plotly({
                "pcd": target_pcd,
                "warp": tree_pcd_complete
            }, center=False)

        anchors = self.canon_mug.to_pcd(mug_param)[self.knns]
        targets_mug = np.mean(anchors + self.deltas, axis=1)
        targets_tree = self.canon_tree.to_pcd(tree_param)[self.target_indices]
        
        # Canonical source obj to canonical target obj.
        trans_cs_to_ct, _, _ = utils.best_fit_transform(targets_mug, targets_tree)

        trans_s_to_b = utils.pos_quat_to_transform(mug_param.position, mug_param.quat)
        trans_t_to_b = utils.pos_quat_to_transform(tree_param.position, tree_param.quat)

        # TODO: Wiggle them out of collision, maybe.
        trans_s_to_t = trans_t_to_b @ trans_cs_to_ct @ np.linalg.inv(trans_s_to_b)
        return trans_s_to_t
