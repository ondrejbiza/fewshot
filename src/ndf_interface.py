import copy as cp
from dataclasses import dataclass, field
from typing import Optional

import time
from matplotlib import transforms
import numpy as np
from numpy.typing import NDArray
import pybullet as pb
import torch
import trimesh
import os
import pickle 
from scipy.spatial.transform import Rotation, Slerp
from typing import List, Optional, Tuple, Union

from src import demo, utils, viz_utils
from src.utils import CanonPart, CanonPartMetadata
from src.pybullet_utils import interpolate, wait_for_interrupt
from src.object_warping import ObjectWarpingSE2Batch, ObjectSE2Batch, ObjectSE3Batch, ObjectWarpingSE3Batch, warp_to_pcd, warp_to_pcd_se2, warp_to_pcd_se3, warp_to_pcd_se3_hemisphere, PARAM_1
from sklearn.decomposition import PCA



@dataclass
class NDFPartInterface:
    """Interface between my method and the Relational Neural Descriptor Fields code."""

    canon_source_path: str = "data/230213_ndf_mugs_scale_large_pca_8_dim_alp_0.01.pkl"
    canon_target_path: str = "data/230213_ndf_trees_scale_large_pca_8_dim_alp_2.pkl"
    canon_source_scale: float = 1.
    canon_target_scale: float = 1.
    source_part_names: list = field(default_factory=lambda: ['cup', 'handle'])
    pcd_subsample_points: Optional[int] = 2000
    nearby_points_delta: float = 0.03
    wiggle: bool = False
    ablate_no_warp: bool = False
    ablate_no_scale: bool = False
    ablate_no_pose_training: bool = False
    ablate_no_size_reg: bool = False

    def __post_init__(self):

        #self.canon_source_parts = utils.CanonPart.from_parts_pickle(self.canon_source_path, self.source_part_names)#utils.CanonPartByParts.from_pickle(self.canon_source_path, self.source_part_names)
        self.canon_target = utils.CanonPart.from_pickle(self.canon_target_path)
        warp_file_stamp = '20240202-160637'
        object_warp_file = f'part_based_warp_models/whole_mug_{warp_file_stamp}'
        cup_warp_file = f'part_based_warp_models/cup_{warp_file_stamp}'
        handle_warp_file = f'part_based_warp_models/handle_{warp_file_stamp}'
        part_canonicals = {}
        whole_object_canonical = pickle.load(open( object_warp_file, 'rb'))
        part_canonicals['cup'] = pickle.load(open( cup_warp_file, 'rb'))
        part_canonicals['handle'] = pickle.load(open( handle_warp_file, 'rb'))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.canon_source_parts = part_canonicals

    def set_demo_info(self, pc_master_dict, demo_idx: int=0, calculate_cost: bool=False, show: bool=True):
        """Process a demonstration."""

        # Get a single demonstration.
        print(self.device)
        source_pcd = pc_master_dict["child"]["demo_start_pcds"][demo_idx]
        source_pcd_parts = pc_master_dict["child"]["demo_start_part_pcds"][demo_idx]
        source_start_part_poses = pc_master_dict["child"]["demo_start_part_poses"][demo_idx]
        self.source_start_part_poses = source_start_part_poses

        source_start = np.array(pc_master_dict["child"]["demo_start_poses"][demo_idx], dtype=np.float64)
        source_final = np.array(pc_master_dict["child"]["demo_final_poses"][demo_idx], dtype=np.float64)
        
        source_start_pos, source_start_quat = source_start[:3], source_start[3:]
        source_final_pos, source_final_quat = source_final[:3], source_final[3:]
        source_start_trans = utils.pos_quat_to_transform(source_start_pos, source_start_quat)
        source_final_trans = utils.pos_quat_to_transform(source_final_pos, source_final_quat)

        source_start_to_final = source_final_trans @ np.linalg.inv(source_start_trans)

        part_start_to_finals = {}

        def get_contact_points(cup_pcl, handle_pcl): 
            knns = utils.get_closest_point_pairs_thresh(cup_pcl, handle_pcl, .00003)#get_closest_point_pairs(cup_pcl, handle_pcl, 11)
            #viz_utils.show_pcds_plotly({'cup': cup_pcl, 'handle':handle_pcl, 'pts_1': cup_pcl[knns[:,1 ]], 'pts_2': handle_pcl[knns[:, 0]]})
            contact_points = np.concatenate([cup_pcl[knns[:,1 ]], handle_pcl[knns[:, 0]]])
            return (knns[:,1 ], knns[:, 0])

        def get_canon_contacts():
            cup_path = './scripts/cup_parts/m1.obj'
            cup_handle_path = './scripts/handles/m1.obj'

            handle_path = './scripts/handles/m4.obj'
            handle_cup_path  = './scripts/cup_parts/m4.obj'

            def load_obj_part(obj_path):
                mesh = utils.trimesh_load_object(obj_path)
                rotation = Rotation.from_euler("zyx", [0., 0., np.pi/2]).as_matrix()
                utils.trimesh_transform(mesh, center=False, scale=None, rotation=rotation)
                ssp = utils.trimesh_create_verts_surface(mesh, num_surface_samples=2000)
                return ssp

            canon_cup, canon_cup_handle = load_obj_part(cup_path), load_obj_part(cup_handle_path)
            canon_handle, canon_handle_cup = load_obj_part(handle_path), load_obj_part(handle_cup_path)
            canon_cup, canon_cup_handle, canon_handle, canon_handle_cup = utils.scale_points_circle([canon_cup, canon_cup_handle, canon_handle, canon_handle_cup], base_scale=0.13)
            canon_contact_cup, canon_contact_cup_handle = get_contact_points(self.canon_source_parts['cup'].canonical_pcl, canon_cup_handle)
            canon_contact_handle_cup, canon_contact_handle = get_contact_points(canon_handle_cup, self.canon_source_parts['handle'].canonical_pcl)

            return canon_contact_cup, canon_contact_handle

        target_pcd = pc_master_dict["parent"]["demo_start_pcds"][demo_idx]
        for part in self.source_part_names:
            if self.pcd_subsample_points is not None and len(source_pcd_parts[part]) > self.pcd_subsample_points:
                source_pcd_parts[part], _ = utils.farthest_point_sample(source_pcd_parts[part], self.pcd_subsample_points)
        
        if self.pcd_subsample_points is not None and len(target_pcd) > self.pcd_subsample_points:
            target_pcd, _ = utils.farthest_point_sample(target_pcd, self.pcd_subsample_points)

        # Perception.
        inference_kwargs = {
            "train_latents": True, #not self.ablate_no_warp,
            "train_scales": True, #not self.ablate_no_scale,
            "train_poses": True, #not self.ablate_no_pose_training
        }

        param_1 = cp.deepcopy(PARAM_1)
        if self.ablate_no_size_reg:
            param_1["object_size_reg"] = 0.

        source_parts = {}
        source_params = {}
        target_pcd_complete = None
        target_param = None

        def cost_batch_pt(source, target): 
            """Calculate the one-sided Chamfer distance between two batches of point clouds in pytorch."""
            # B x N x K
            diff = torch.sqrt(torch.sum(torch.square(source[:, :, None] - target[:, None, :]), dim=3))
            diff_flat = diff.view(diff.shape[0] * diff.shape[1], diff.shape[2])
            c_flat = diff_flat[list(range(len(diff_flat))), torch.argmin(diff_flat, dim=1)]
            c = c_flat.view(diff.shape[0], diff.shape[1])
            return torch.mean(c, dim=1)

        def contact_constraint(source, target, source_contact_indices, canon_points, weight=10):  
            constraint = cost_batch_pt(canon_points, source[:, source_contact_indices, :]) * weight
            return cost_batch_pt(source, target) + constraint

        source_contacts = {}
        canon_contacts = {}
        source_contacts['cup'], source_contacts['handle'] = get_contact_points(source_pcd_parts['cup'], source_pcd_parts['handle'])
        canon_contacts['cup'], canon_contacts['handle'] = get_canon_contacts()
        #Source part warping
        for part in self.source_part_names:
            if part == 'cup':
                n_angles = 12
            else:
                n_angles = 24
            cost_function = lambda source, target, canon_points: contact_constraint(source, target, source_contacts[part], canon_points, weight=1)

            # warp = ObjectSE2Batch(
            #     self.canon_source_parts[part], source_pcd_parts[part], 'cpu', **param_1,
            #     init_scale=self.canon_source_scale) 
            # source_parts[part], _, source_params[part] = warp_to_pcd_se2 (warp, n_angles=18, n_batches=1, inference_kwargs=inference_kwargs)
            
            warp = ObjectWarpingSE3Batch(self.canon_source_parts[part], source_pcd_parts[part], self.device, **cp.deepcopy(PARAM_1),) 
            source_parts[part], _, source_params[part] = warp_to_pcd_se3(warp, n_angles, n_batches=3, inference_kwargs=inference_kwargs)
            # warp = ObjectWarpingSE2Batch(
            #     self.canon_source_parts[part], source_pcd_parts[part], 'cpu', **param_1,
            #     init_scale=self.canon_source_scale) 
            # source_parts[part], _, source_params[part] = warp_to_pcd_se2 (warp, n_angles=18, n_batches=1, inference_kwargs=inference_kwargs)
           
            viz_utils.show_pcds_plotly({'canon': source_pcd_parts[part], 'source':source_parts[part]})
            # viz_utils.show_pcds_plotly({'warp': self.canon_source_parts[part].to_transformed_pcd(source_params[part]),
            #                             'warped_contacts': self.canon_source_parts[part].to_transformed_pcd(source_params[part])[canon_contacts[part]], 
            #                             'orig': source_pcd_parts[part],
            #                             'orig_contacts': source_pcd_parts[part][source_contacts[part]]})


        warp = ObjectWarpingSE2Batch(
            self.canon_target, target_pcd, self.device, **param_1,
            init_scale=self.canon_target_scale)
        target_pcd_complete, _, target_param = warp_to_pcd_se2(warp, n_angles=12, n_batches=1, inference_kwargs=inference_kwargs)

        state_dict = {'source_parts': source_parts, 'source_params':source_params,\
                      'target_pcd_complete': target_pcd_complete, 'target_param': target_param}

        if show:
            None
            # viz_utils.show_pcds_plotly({
            #     "pcd": source_pcd,
            #     "warp": source_pcd_complete
            # }, center=False)
            

            viz_utils.show_pcds_plotly({
                "pcd": source_pcd,
                "target_cup": source_pcd_parts['cup'],
                "target_handle": source_pcd_parts['handle'],
                "cup": source_parts['cup'],
                "handle": source_parts['handle'],
                #"warp": target_pcd
            }, center=False)

        # adjust to be part_based
        # Move object to final pose.
        warped_source_meshes = {}
        warped_source_part_pcds = {}
        for part in self.source_part_names:
            trans = utils.pos_quat_to_transform(source_params[part].position, source_params[part].quat)
            trans = source_start_to_final @ trans
            pos, quat = utils.transform_to_pos_quat(trans)
            source_params[part].position = pos
            source_params[part].quat = quat

            warped_source_part_pcds[part] = self.canon_source_parts[part].to_transformed_pcd(source_params[part])
            warped_source_meshes[part] = self.canon_source_parts[part].to_transformed_mesh(source_params[part])

        mesh = trimesh.util.concatenate([warped_source_meshes[part] for part in self.source_part_names])
        mesh.export(f"tmp_source.obj")
        utils.convex_decomposition(mesh, f"tmp_source_cd.obj")

        mesh = self.canon_target.to_mesh(target_param)
        mesh.export("tmp_target.stl")
        utils.convex_decomposition(mesh, "tmp_target.obj")

        #combine the meshes

        # Add predicted meshes to pybullet.
        source_pb = pb.loadURDF("tmp_source.urdf", useFixedBase=True)
        pb.resetBasePositionAndOrientation(source_pb, source_final_pos, source_final_quat)

        target_pb = pb.loadURDF("tmp_target.urdf", useFixedBase=True)
        pb.resetBasePositionAndOrientation(target_pb, target_param.position, target_param.quat)

        # print("LEARNED_TRANS")
        # print(target_param.position)
        # print(target_param.quat)

        # print("TARGET START")
        # print(pc_master_dict["parent"]["demo_start_poses"][demo_idx])

        # print("TARGET END")
        # print(pc_master_dict["parent"]["demo_final_poses"][demo_idx])

        # Save nearby points.
        self.knns, self.deltas, self.target_indices = demo.save_place_nearby_points_by_parts_v2(
            self.source_part_names, self.canon_source_parts, source_params, self.canon_target,
            target_param, self.nearby_points_delta)

        targets_source = {}
        for part in self.source_part_names:
            anchors = source_parts[part][self.knns[part]]
            targets_source[part] = np.mean(anchors + self.deltas[part], axis=1)
        
        # Remove predicted meshes from pybullet.
        pb.removeBody(source_pb)
        pb.removeBody(target_pb)

        if calculate_cost:
            print("CALCULATING COST")

            # Make a prediction based on the training sample and calculate the distance between it and the ground-truth.
            trans_predicted = self.infer_relpose(source_pcd_parts, target_pcd, se3=True)
            cost = utils.pose_distance(trans_predicted, source_start_to_final)

            return cost


    def infer_relpose(self, source_pcds, target_pcd, se3: bool=False, show: bool=True):

        for part in self.source_part_names:
            """Make prediction about the final pose of the source object."""
            if self.pcd_subsample_points is not None and len(source_pcds[part]) > self.pcd_subsample_points:
                source_pcds[part], _ = utils.farthest_point_sample(source_pcds[part], self.pcd_subsample_points)
        if self.pcd_subsample_points is not None and len(target_pcd) > self.pcd_subsample_points:
                target_pcd, _ = utils.farthest_point_sample(target_pcd, self.pcd_subsample_points)

        inference_kwargs = {
            "train_latents": True, #not self.ablate_no_warp,
            "train_scales": True, #not self.ablate_no_scale,
            "train_poses": True, #not self.ablate_no_pose_training
        }

        param_1 = cp.deepcopy(PARAM_1)
        if self.ablate_no_size_reg:
            param_1["object_size_reg"] = 0.

        source_parts_complete = {}
        source_params = {}
        for part in self.source_part_names: 

            #cost_function = lambda source, target, canon_points: contact_constraint(source, target, source_contacts[part], canon_points, weight=1)

            if part == 'cup':
                n_angles = 12

            else:
                n_angles = 24

            if se3:
                warp = ObjectWarpingSE3Batch(
                    self.canon_source_parts[part], source_pcds[part], self.device, **param_1,
                    init_scale=self.canon_source_scale)
                source_parts_complete[part], _, source_params[part] = warp_to_pcd_se3_hemisphere(warp, n_angles=n_angles, n_batches=3, inference_kwargs=inference_kwargs)
            else:
                warp = ObjectWarpingSE2Batch(
                    self.canon_source_parts[part], source_pcds[part], self.device, **param_1,
                    init_scale=self.canon_source_scale)
                source_parts_complete[part], _, source_params[part] = warp_to_pcd_se2(warp, n_angles=n_angles, n_batches=3, inference_kwargs=inference_kwargs)

        warp = ObjectWarpingSE2Batch(
            self.canon_target, target_pcd, self.device, **param_1,
            init_scale=self.canon_target_scale)
        target_pcd_complete, _, target_param = warp_to_pcd_se2(warp, n_angles=12, n_batches=1, inference_kwargs=inference_kwargs)
        
        state_dict = {'source_parts': source_parts_complete, 'source_params':source_params,\
                      'target_pcd_complete': target_pcd_complete, 'target_param': target_param}

        source_pcd = np.concatenate([source_pcds[part] for part in self.source_part_names])
        source_pcd_complete = np.concatenate([source_parts_complete[part] for part in self.source_part_names])

        meshes = {}
        canon_pcds = {}
        transformed_canon_pcds = {}
        for part in self.source_part_names:
            trans = utils.pos_quat_to_transform(source_params[part].position, source_params[part].quat)

            pos, quat = utils.transform_to_pos_quat(trans)
            source_params[part].position = pos
            source_params[part].quat = quat
            canon_pcds[part] = self.canon_source_parts[part].to_pcd(source_params[part])
            transformed_canon_pcds[part] = self.canon_source_parts[part].to_transformed_pcd(source_params[part])
            meshes[part] = self.canon_source_parts[part].to_transformed_mesh(source_params[part])

        source_mesh = trimesh.util.concatenate([meshes[part] for part in self.source_part_names])
        canon_source = np.concatenate([transformed_canon_pcds[part] for part in self.source_part_names])

        # if show:
        #     viz_utils.show_pcds_plotly({
        #         "pcd": source_pcd,
        #         "warp": source_pcd_complete
        #     }, center=False)
        #     viz_utils.show_pcds_plotly({
        #         "pcd": target_pcd,
        #         "warp": target_pcd_complete
        #     }, center=False)

        targets_source = {key:[] for key in self.source_part_names}
        targets_target = {key:[] for key in self.source_part_names}
        trans_cs_to_ct = {}
        for part in self.source_part_names:
            anchors = canon_pcds[part][self.knns[part]]
            targets_source[part].append(np.mean(anchors + self.deltas[part], axis=1))
            targets_target[part].append(self.canon_target.to_pcd(target_param)[self.target_indices[part]])
            trans_cs_to_ct[part], _, _ = utils.best_fit_transform(np.array(targets_source[part]).squeeze(), np.array(targets_target[part]).squeeze())

        trans_s_to_b = {}#utils.pos_quat_to_transform(np.mean(source_pcd_complete, axis=0) - np.mean(source_pcd, axis=0), [0,0,0,1]) @ trans_cs_to_ct#utils.pos_quat_to_transform(source_params[part].position, source_params[part].quat)
        for part in self.source_part_names: 
            trans_s_to_b[part] = utils.pos_quat_to_transform(source_params[part].position, source_params[part].quat)
        trans_t_to_b = utils.pos_quat_to_transform(target_param.position, target_param.quat)

        trans_s_to_t = {}
        for part in self.source_part_names:
            trans_s_to_t[part] = trans_t_to_b @ trans_cs_to_ct[part] @ np.linalg.inv(trans_s_to_b[part])
        

        viz_utils.show_pcds_plotly({
            # "cup": canon_pcds['cup'], 
            # "handle": canon_pcds['handle'],
            "cup_moved": utils.transform_pcd(source_pcds['cup'], trans_s_to_t['cup']), 
            "handle_moved": utils.transform_pcd(source_pcds['handle'], trans_s_to_t['handle']),
            "target": target_pcd, #self.canon_target.to_transformed_pcd(target_param),  
            "anchors_cup": utils.transform_pcd(utils.transform_pcd(np.concatenate(targets_source['cup']), trans_s_to_b['cup']), trans_s_to_t['cup']),
            "anchors_handle": utils.transform_pcd(utils.transform_pcd(np.concatenate(targets_source['handle']), trans_s_to_b['handle']), trans_s_to_t['handle']),
            #"anchors_t": targets_target,
        }, center=False)

        component_pcls = []
        component_mesh_vertices = []
        component_mesh_faces = []

        def subsample(pcl, num_samples=1000):
            """Randomly subsample the canonical object, including its PCA projection."""
            indices = np.random.randint(0, pcl.shape[0], num_samples)
            subsampled_pcl = pcl[indices]
            return subsampled_pcl

        for part in self.source_part_names:
            component_pcls.append(utils.transform_pcd(source_pcds[part], trans_s_to_t[part]))
            component_mesh_vertices.append(meshes[part].vertices)
            component_mesh_faces.append(meshes[part].faces + sum([len(pcl) for pcl in component_mesh_vertices[:-1]]))

        canon_pcl = np.concatenate(component_pcls, axis=0)
        print(canon_pcl.shape)

        canon_mesh_vertices = np.concatenate(component_mesh_vertices, axis=0)
        canon_mesh_faces = np.concatenate(component_mesh_faces, axis=0)
        center_transform = utils.pos_quat_to_transform(np.mean(canon_pcl, axis=0), np.array([0., 0., 0., 1.]))
        metadata = CanonPartMetadata("none", "none", ["none"], None)
        combined_part = CanonPart(canon_pcl, canon_mesh_vertices, canon_mesh_faces, center_transform, metadata, None, None)
        
        pickle.dump(combined_part, open('temp_combined.pkl', 'wb'))

        final_inference_kwargs = {
            "train_latents": False,
            "train_scales": False,
            "train_poses": True
        }

        #furthest point sample source
        #get both centered pointclouds
        centered_combined = utils.center_pcl(combined_part.canonical_pcl)
        real_combined = cp.deepcopy(combined_part.canonical_pcl)
        source_downsampled = utils.center_pcl(utils.farthest_point_sample(source_pcd, 1000)[0])
        combined_part.canonical_pcl = centered_combined

        combined_warp = ObjectSE3Batch(
                    combined_part, source_downsampled, self.device, **param_1, init_scale=1)
        combined_complete, _, combined_params = warp_to_pcd_se3(combined_warp, n_angles=12, n_batches=3, inference_kwargs=final_inference_kwargs)
        final_transform = utils.pos_quat_to_transform(np.mean(real_combined, axis=0), (0,0,0,1)) @ \
                          np.linalg.inv(utils.pos_quat_to_transform(combined_params.position, combined_params.quat)) @ \
                          utils.pos_quat_to_transform(-np.mean(source_pcd, axis=0), (0,0,0,1))
                          
                          #   \

        combined_part.canonical_pcl = real_combined

        viz_utils.show_pcds_plotly({'pcd':source_pcd,
                                    'canon_registration': canon_pcl,
                                    '.pcl': combined_part.canonical_pcl,
                                    'untransformed': combined_part.to_pcd(combined_params),
                                    'transform_registration': combined_part.to_transformed_pcd(combined_params),
                                    'trans_pcd':utils.transform_pcd(source_pcd, final_transform),
                                    'centered??': utils.center_pcl(source_pcd),
                                    #'t2_trans_pcd':utils.transform_pcd(source_pcd, np.linalg.inv(utils.pos_quat_to_transform(combined_params.position, combined_params.quat))),
                                    'target':self.canon_target.to_transformed_pcd(target_param)})

        # Canonical source obj to canonical target obj.
        
        # viz_utils.show_pcds_plotly({
        #     "pcd": utils.transform_pcd(canon_source, trans_cs_to_ct), "anchors": utils.transform_pcd(targets_source, trans_cs_to_ct), "anchors_t": targets_target,
        # }, center=False)
        # exit(0)

        # viz_utils.show_pcds_plotly({
        #     "cup": canon_pcds['cup'], 
        #     "handle": canon_pcds['handle'], 
        #     "transformed_cup": transformed_canon_pcds['cup'],
        #     "transformed_handle": transformed_canon_pcds['handle'], 
        # }, center=False)
        # exit(0)


        # # Save the mesh and its convex decomposition.
        # source_mesh.export("tmp_source.obj")
        # utils.convex_decomposition(source_mesh, "tmp_source_cd.obj")

        # mesh = self.canon_target.to_mesh(target_param)
        # mesh.export("tmp_target.stl")
        # utils.convex_decomposition(mesh, "tmp_target.obj")

        # # Add predicted meshes to pybullet.
        # source_pb = pb.loadURDF("tmp_source.urdf", useFixedBase=True)
        # pb.resetBasePositionAndOrientation(source_pb, *utils.transform_to_pos_quat(source_start_trans))

        # target_pb = pb.loadURDF("tmp_target.urdf", useFixedBase=True)
        # pb.resetBasePositionAndOrientation(target_pb, *utils.transform_to_pos_quat(trans_t_to_b))

        # if self.wiggle:
        #     # Wiggle the source object out of collision.
        #     src_pos, src_quat = utils.wiggle(source_pb, target_pb)
        #     trans_s_to_b = utils.pos_quat_to_transform(src_pos, src_quat)

        # # # Remove predicted meshes from pybullet.
        # pb.removeBody(source_pb)
        # pb.removeBody(target_pb)

        return final_transform #trans_s_to_t


@dataclass
class NDFInterface:
    """Interface between my method and the Relational Neural Descriptor Fields code."""

    canon_source_path: str = "data/230213_ndf_mugs_scale_large_pca_8_dim_alp_0.01.pkl"
    canon_target_path: str = "data/230213_ndf_trees_scale_large_pca_8_dim_alp_2.pkl"
    canon_source_scale: float = 1.
    canon_target_scale: float = 1.
    pcd_subsample_points: Optional[int] = 2000
    nearby_points_delta: float = 0.03
    wiggle: bool = False
    ablate_no_warp: bool = False
    ablate_no_scale: bool = False
    ablate_no_pose_training: bool = False
    ablate_no_size_reg: bool = False

    def __post_init__(self):
        self.canon_source = utils.CanonPart.from_pickle(self.canon_source_path)
        self.canon_target = utils.CanonPart.from_pickle(self.canon_target_path)

    def set_demo_info(self, pc_master_dict, demo_idx: int=0, calculate_cost: bool=False, show: bool=True):
        """Process a demonstration."""

        # Get a single demonstration.
        source_pcd = pc_master_dict["child"]["demo_start_pcds"][demo_idx]
        source_start = np.array(pc_master_dict["child"]["demo_start_poses"][demo_idx], dtype=np.float64)
        source_final = np.array(pc_master_dict["child"]["demo_final_poses"][demo_idx], dtype=np.float64)

        source_start_pos, source_start_quat = source_start[:3], source_start[3:]
        source_final_pos, source_final_quat = source_final[:3], source_final[3:]
        source_start_trans = utils.pos_quat_to_transform(source_start_pos, source_start_quat)
        self.source_start_trans = source_start_trans
        source_final_trans = utils.pos_quat_to_transform(source_final_pos, source_final_quat)
        source_start_to_final = source_final_trans @ np.linalg.inv(source_start_trans)

        target_pcd = pc_master_dict["parent"]["demo_start_pcds"][demo_idx]
        if self.pcd_subsample_points is not None and len(source_pcd) > self.pcd_subsample_points:
            source_pcd, _ = utils.farthest_point_sample(source_pcd, self.pcd_subsample_points)
        if self.pcd_subsample_points is not None and len(target_pcd) > self.pcd_subsample_points:
            target_pcd, _ = utils.farthest_point_sample(target_pcd, self.pcd_subsample_points)

        # Perception.
        inference_kwargs = {
            "train_latents": True,#not self.ablate_no_warp,
            "train_scales": True, #not self.ablate_no_scale,
            "train_poses": True,#not self.ablate_no_pose_training
        }

        param_1 = cp.deepcopy(PARAM_1)
        if self.ablate_no_size_reg:
            param_1["object_size_reg"] = 0.

        warp = ObjectWarpingSE2Batch(
            self.canon_source, [], source_pcd, 'cpu', cost_function=None, **param_1,
            init_scale=self.canon_source_scale)
        source_pcd_complete, _, source_param = warp_to_pcd_se2(warp, n_angles=12, n_batches=1, inference_kwargs=inference_kwargs)

        warp = ObjectWarpingSE2Batch(
            self.canon_target, [], target_pcd, 'cpu', cost_function=None, **param_1,
            init_scale=self.canon_target_scale)
        target_pcd_complete, _, target_param = warp_to_pcd_se2(warp, n_angles=12, n_batches=1, inference_kwargs=inference_kwargs)

        if show:
            # viz_utils.show_pcds_plotly({
            #     "pcd": source_pcd,
            #     "warp": source_pcd_complete
            # }, center=False)
            # viz_utils.show_pcds_plotly({
            #     "pcd": target_pcd,
            #     "warp": target_pcd_complete
            # }, center=False)

            viz_utils.show_pcds_plotly({
                "pcd": source_pcd,
                "source_pcd": source_pcd_complete, 
                "warp": target_pcd
            }, center=False)

        # Move object to final pose.
        trans = utils.pos_quat_to_transform(source_param.position, source_param.quat)



        trans = source_start_to_final @ trans
        pos, quat = utils.transform_to_pos_quat(trans)
        source_param.position = pos
        source_param.quat = quat

       #viz_utils.show_pcds_plotly({'pcd':self.canon_source.to_transformed_pcd(source_param)})
        #exit(0)


        # Save the mesh and its convex decomposition.
        mesh = self.canon_source.to_mesh(source_param)
        mesh.export("tmp_source.stl")
        utils.convex_decomposition(mesh, "tmp_source.obj")

        mesh = self.canon_target.to_mesh(target_param)
        mesh.export("tmp_target.stl")
        utils.convex_decomposition(mesh, "tmp_target.obj")

        # Add predicted meshes to pybullet.
        source_pb = pb.loadURDF("tmp_source.urdf", useFixedBase=True)
        pb.resetBasePositionAndOrientation(source_pb, source_param.position, source_param.quat)

        target_pb = pb.loadURDF("tmp_target.urdf", useFixedBase=True)
        pb.resetBasePositionAndOrientation(target_pb, target_param.position, target_param.quat)

        time.sleep(10.0)

        # Save nearby points.
        self.knns, self.deltas, self.target_indices = demo.save_place_nearby_points_v2(
            source_pb, target_pb, self.canon_source, source_param, self.canon_target,
            target_param, self.nearby_points_delta)


        anchors = source_pcd_complete[self.knns]
        targets_source = np.mean(anchors + self.deltas, axis=1)
        # viz_utils.show_pcds_plotly({
        #         "pcd": source_pcd,
        #         "target": target_pcd,
        #         "indices":targets_source, 
        #         #"warp": target_pcd
        #     }, center=False)

        # Remove predicted meshes from pybullet.
        pb.removeBody(source_pb)
        pb.removeBody(target_pb)

        if calculate_cost:
            # Make a prediction based on the training sample and calculate the distance between it and the ground-truth.
            trans_predicted = self.infer_relpose(source_pcd, target_pcd)
            
            viz_utils.show_pcds_plotly({
                "source_pcd": utils.transform_pcd(source_pcd, trans_predicted),
                "source_pcd_gt": utils.transform_pcd(source_pcd, source_start_to_final),
                "target_pcd": target_pcd,
                })
            print("STARTING POS")
            print(source_start_pos)
            print("FINAL POS")
            print(source_final_pos)

            return utils.pose_distance(trans_predicted, source_start_to_final)

    def infer_relpose(self, source_pcd, target_pcd, se3: bool=False, show: bool=True):
        """Make prediction about the final pose of the source object."""
        if self.pcd_subsample_points is not None and len(source_pcd) > self.pcd_subsample_points:
            source_pcd, _ = utils.farthest_point_sample(source_pcd, self.pcd_subsample_points)
        if self.pcd_subsample_points is not None and len(target_pcd) > self.pcd_subsample_points:
            target_pcd, _ = utils.farthest_point_sample(target_pcd, self.pcd_subsample_points)

        inference_kwargs = {
            "train_latents": not self.ablate_no_warp,
            "train_scales": not self.ablate_no_scale,
            "train_poses": not self.ablate_no_pose_training
        }

        param_1 = cp.deepcopy(PARAM_1)
        if self.ablate_no_size_reg:
            param_1["object_size_reg"] = 0.

        if se3:
            warp = ObjectWarpingSE3Batch(
                self.canon_source, [], source_pcd, 'cpu', cost_function=None, **param_1,
                init_scale=self.canon_source_scale)
            source_pcd_complete, _, source_param = warp_to_pcd_se3(warp, n_angles=12, n_batches=15, inference_kwargs=inference_kwargs)
        else:
            warp = ObjectWarpingSE2Batch(
                self.canon_source, [], source_pcd, 'cpu', cost_function=None, **param_1,
                init_scale=self.canon_source_scale)
            source_pcd_complete, _, source_param = warp_to_pcd_se2(warp, n_angles=12, n_batches=1, inference_kwargs=inference_kwargs)

        warp = ObjectWarpingSE2Batch(
            self.canon_target, [], target_pcd, 'cpu', cost_function=None, **param_1,
            init_scale=self.canon_target_scale)
        target_pcd_complete, _, target_param = warp_to_pcd_se2(warp, n_angles=12, n_batches=1, inference_kwargs=inference_kwargs)

        if show:
            viz_utils.show_pcds_plotly({
                "pcd": source_pcd,
                "warp": source_pcd_complete
            }, center=False)
            viz_utils.show_pcds_plotly({
                "pcd": target_pcd,
                "warp": target_pcd_complete
            }, center=False)

        # Save nearby points.

        anchors = self.canon_source.to_pcd(source_param)[self.knns]
        targets_source = np.mean(anchors + self.deltas, axis=1)
        targets_target = self.canon_target.to_pcd(target_param)[self.target_indices]
        viz_utils.show_pcds_plotly({'pcd': self.canon_source.to_pcd(source_param), 'anchors': targets_source})
        #input("ready_to_continue?")

        # viz_utils.show_pcds_plotly({
        #     "pcd": self.canon_source.to_pcd(source_param), "anchors": targets_source, "anchors_t": targets_target
        # }, center=False)

        # Canonical source obj to canonical target obj.
        trans_cs_to_ct, _, _ = utils.best_fit_transform(targets_source, targets_target)

        trans_s_to_b = utils.pos_quat_to_transform(source_param.position, source_param.quat)

        trans_t_to_b = utils.pos_quat_to_transform(target_param.position, target_param.quat)

        # Save the mesh and its convex decomposition.
        mesh = self.canon_source.to_mesh(source_param)
        mesh.export("tmp_source.stl")
        utils.convex_decomposition(mesh, "tmp_source.obj")

        mesh = self.canon_target.to_mesh(target_param)
        mesh.export("tmp_target.stl")
        utils.convex_decomposition(mesh, "tmp_target.obj")

        # Add predicted meshes to pybullet.
        source_pb = pb.loadURDF("tmp_source.urdf", useFixedBase=True)
        pb.resetBasePositionAndOrientation(source_pb, *utils.transform_to_pos_quat(trans_s_to_b))

        target_pb = pb.loadURDF("tmp_target.urdf", useFixedBase=True)
        pb.resetBasePositionAndOrientation(target_pb, *utils.transform_to_pos_quat(trans_t_to_b))

        if self.wiggle:
            # Wiggle the source object out of collision.
            src_pos, src_quat = utils.wiggle(source_pb, target_pb)
            trans_s_to_b = utils.pos_quat_to_transform(src_pos, src_quat)


        # Remove predicted meshes from pybullet.
        pb.removeBody(source_pb)
        pb.removeBody(target_pb)

        # Compute relative transform.
        trans_s_to_t = trans_t_to_b @ trans_cs_to_ct @ np.linalg.inv(trans_s_to_b)
        return trans_s_to_t
