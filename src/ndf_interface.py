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
from src.object_warping import (
    ObjectWarpingSE2Batch,
    ObjectSE2Batch,
    ObjectSE3Batch,
    ObjectWarpingSE3Batch,
    warp_to_pcd,
    warp_to_pcd_se2,
    warp_to_pcd_se3,
    warp_to_pcd_se3_hemisphere,
    PARAM_1,
    ALIGNMENT_PARAM,
    mask_and_cost_batch_pt,
)
from sklearn.decomposition import PCA
import itertools


# def optimize_plane():
#     cost = #distance of point from plane and


def get_part_labels(part_pairs):
    part_labels = {}
    for part_pair in part_pairs:

        ordered_part_names = list(part_pair.keys())
        ordered_part_names.sort()

        dists = np.sum(
            np.square(part_pair[ordered_part_names[0]][None] - part_pair[ordered_part_names[1]][:, None]),
            axis=-1,
        )
        
        part_dists = {ordered_part_names[i]: np.min(dists, axis=i) for i in range(len(ordered_part_names))} #np.min
        for part in ordered_part_names:
            if part not in part_labels.keys():
                part_labels[part] = []
            min_dist = np.min(part_dists[part])
            part_labels[part].append(np.where(
                    part_dists[part]-min_dist < np.mean(part_dists[part]-min_dist) * .6,
                    np.zeros_like(part_dists[part]),
                    np.ones_like(part_dists[part]),
                )
            )
    return part_labels


def get_canon_labels(part_pairs, part_canonicals, part_names):

    if part_pairs is None:
        part_pairs = [{part_1: part_canonicals[part_1], part_2: part_canonicals[part_2]} for part_1, part_2 in itertools.combinations(part_names, r=2) if part_1 != part_2]

    # Doing adjustment to the centered/scaled parts to accurately approximate these labels
    part_adjustment = {
        part: utils.pos_quat_to_transform(
            part_canonicals[part].center_transform, (0, 0, 0, 1)
        )
        for part in part_names
    }


    ten_scaled_parts = utils.scale_points_circle(
        [part_canonicals[part].canonical_pcl for part in part_names], base_scale=10
    )

    adjusted_part_canon = {
        part_names[i]: utils.transform_pcd(ten_scaled_parts[i], part_adjustment[part_names[i]])
        for i in range(len(part_names))
    }

    contact_parts = utils.scale_points_circle(
        [adjusted_part_canon[part] for part in part_names], base_scale=0.1
    )

    #TODO remove hackiness
    if part_names[0] == 'body':
        contact_parts[0] = utils.scale_points_circle([contact_parts[0]], base_scale=.075)[0]

    #Recreating the part pairs with the correct relative poses
    contact_pairs = []
    for pair in part_pairs:
        contact_pairs.append({p: contact_parts[part_names.index(p)] for p in pair.keys()})
        #viz_utils.show_pcds_plotly({p: contact_parts[part_names.index(p)] for p in pair.keys()}).show()

    canon_labels = get_part_labels(
        contact_pairs,
    )  # part_names)
    return canon_labels


def get_z_descriptors(part_pcls, part_names):
    part_labels = {part: [] for part in part_names}
    for part in part_names: 
        z_mean = np.mean(part_pcls[part][:,2])
        
        part_labels[part].append(np.where(
                part_pcls[part][:,2] > z_mean,
                np.zeros_like(part_pcls[part][:, 0]),
                np.ones_like(part_pcls[part][:, 0]),
            ))
    return part_labels

    # n_descriptors = 1
    # part_pcas = {}
    # part_labels = {part: [] for part in part_names}
    # for part in part_names: 
    #     part_pcas[part] = PCA(n_components=2)
    #     components = part_pcas[part].fit_transform(part_pcls[part]).T
    #     for i in range(n_descriptors):
    #         component_mean = np.mean(components[i, :])
    #         part_labels[part].append(np.where(
    #                 components[i,:] > component_mean,
    #                 np.zeros_like(part_pcls[part][:, 0]),
    #                 np.ones_like(part_pcls[part][:, 0]),
    #             ))
    # return part_labels

def get_whole_alignment(child, child_mesh_faces, child_mesh_vertices, trans_s_to_t):
    # Building the final alignment constraint pointcloud
    canon_pcl = utils.transform_pcd(child, trans_s_to_t)
    canon_mesh_vertices = child_mesh_vertices
    canon_mesh_faces = child_mesh_faces
    center_transform = utils.pos_quat_to_transform(
        np.mean(canon_pcl, axis=0), np.array([0.0, 0.0, 0.0, 1.0])
    )
    metadata = CanonPartMetadata("none", "none", ["none"], None)
    combined_part = CanonPart(
        canon_pcl,
        canon_mesh_vertices,
        canon_mesh_faces,
        center_transform,
        metadata,
        None,
        None,
    )

    final_inference_kwargs = {
        "train_latents": False,
        "train_scales": False,
        "train_poses": True,
    }

    centered_combined = utils.center_pcl(combined_part.canonical_pcl)
    real_combined = cp.deepcopy(combined_part.canonical_pcl)

    source_downsampled, source_downsampled_indices = utils.farthest_point_sample(
        source_pcd, 1000
    )
    source_downsampled = utils.center_pcl(source_downsampled)
    combined_part.canonical_pcl = centered_combined

    cup_label = np.zeros(component_pcls[0].shape[0])
    handle_label = np.ones(component_pcls[1].shape[0])
    canon_part_labels = np.concatenate([cup_label, handle_label])

    if target_labels is not None:
        cost_function = (
            lambda source, target, canon_part_labels: mask_and_cost_batch_pt(
                source,
                canon_part_labels,
                target,
                target_labels[source_downsampled_indices],
            )
        )

        combined_warp = ObjectSE3Batch(
            combined_part,
            source_downsampled,
            self.device,
            canon_labels=canon_part_labels,
            cost_function=cost_function,
            **alignment_param,
            init_scale=1,
        )
    else:
        combined_warp = ObjectSE3Batch(
            combined_part,
            source_downsampled,
            self.device,
            **alignment_param,
            init_scale=1,
        )

    combined_complete, combined_costs, combined_params = warp_to_pcd_se3(
        combined_warp,
        n_angles=15,
        n_batches=12,
        inference_kwargs=final_inference_kwargs,
    )
    final_transform = (
        utils.pos_quat_to_transform(np.mean(real_combined, axis=0), (0, 0, 0, 1))
        @ np.linalg.inv(
            utils.pos_quat_to_transform(combined_params.position, combined_params.quat)
        )
        @ utils.pos_quat_to_transform(-np.mean(source_pcd, axis=0), (0, 0, 0, 1))
    )

    combined_part.canonical_pcl = real_combined


def get_part_alignment():
    component_pcls = []
    component_mesh_vertices = []
    component_mesh_faces = []

    for part in self.source_part_names:
        component_pcls.append(
            utils.transform_pcd(source_pcds[part], trans_s_to_t[part])
        )
        component_mesh_vertices.append(meshes[part].vertices)
        component_mesh_faces.append(
            meshes[part].faces + sum([len(pcl) for pcl in component_mesh_vertices[:-1]])
        )

    # Building the final alignment constraint pointcloud
    canon_pcl = np.concatenate(component_pcls, axis=0)
    canon_mesh_vertices = np.concatenate(component_mesh_vertices, axis=0)
    canon_mesh_faces = np.concatenate(component_mesh_faces, axis=0)
    center_transform = utils.pos_quat_to_transform(
        np.mean(canon_pcl, axis=0), np.array([0.0, 0.0, 0.0, 1.0])
    )
    metadata = CanonPartMetadata("none", "none", ["none"], None)
    combined_part = CanonPart(
        canon_pcl,
        canon_mesh_vertices,
        canon_mesh_faces,
        center_transform,
        metadata,
        None,
        None,
    )

    final_inference_kwargs = {
        "train_latents": False,
        "train_scales": False,
        "train_poses": True,
    }

    centered_combined = utils.center_pcl(combined_part.canonical_pcl)
    real_combined = cp.deepcopy(combined_part.canonical_pcl)

    source_downsampled, source_downsampled_indices = utils.farthest_point_sample(
        source_pcd, 1000
    )
    source_downsampled = utils.center_pcl(source_downsampled)
    combined_part.canonical_pcl = centered_combined

    # cup_label = np.zeros(component_pcls[0].shape[0])
    # handle_label = np.ones(component_pcls[1].shape[0])
    canon_part_labels = np.array(
        itertools.chain(
            *[
                [i for _ in component_pcls[j].shape[0]]
                for i, j in enumerate(list(range(len(component_pcls))))
            ]
        )
    )  # np.concatenate([cup_label, handle_label])

    if target_labels is not None:
        cost_function = (
            lambda source, target, canon_part_labels: mask_and_cost_batch_pt(
                source,
                canon_part_labels,
                target,
                target_labels[source_downsampled_indices],
            )
        )

        combined_warp = ObjectSE3Batch(
            combined_part,
            source_downsampled,
            self.device,
            canon_labels=canon_part_labels,
            cost_function=cost_function,
            **alignment_param,
            init_scale=1,
        )
    else:
        combined_warp = ObjectSE3Batch(
            combined_part,
            source_downsampled,
            self.device,
            **alignment_param,
            init_scale=1,
        )

    combined_complete, combined_costs, combined_params = warp_to_pcd_se3(
        combined_warp,
        n_angles=15,
        n_batches=12,
        inference_kwargs=final_inference_kwargs,
    )
    final_transform = (
        utils.pos_quat_to_transform(np.mean(real_combined, axis=0), (0, 0, 0, 1))
        @ np.linalg.inv(
            utils.pos_quat_to_transform(combined_params.position, combined_params.quat)
        )
        @ utils.pos_quat_to_transform(-np.mean(source_pcd, axis=0), (0, 0, 0, 1))
    )

    combined_part.canonical_pcl = real_combined



@dataclass
class NDFPartInterface:
    """Interface between my method and the Relational Neural Descriptor Fields code."""

    canon_source_parts_paths: dict = (field(default_factory=lambda: {0: None}),)
    canon_target_parts_paths: dict = (field(default_factory=lambda: {0: None}),)
    canon_source_scale: float = 1.0
    canon_target_scale: float = 1.0
    source_part_names: list = field(default_factory=lambda: ["cup", "handle"])
    target_part_names: list = field(default_factory=lambda: ["trunk", "branch"])
    pcd_subsample_points: Optional[int] = 2000
    nearby_points_delta: float = 0.03
    wiggle: bool = False
    ablate_no_warp: bool = False
    ablate_no_scale: bool = False
    ablate_no_pose_training: bool = False
    ablate_no_size_reg: bool = False

    def __post_init__(self):
        self.canon_source_parts = {
            part: CanonPart.from_pickle(self.canon_source_parts_paths[part])
            for part in self.source_part_names
        }
        self.canon_target_parts = {
            part: CanonPart.from_pickle(self.canon_target_parts_paths[part])
            for part in self.target_part_names
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.target_part_names[0] == "syn_rack_med_1_trunk":
            self.target_part_names = ["trunk", "branch"]
            self.canon_target_parts = {
                "trunk": self.canon_target_parts["syn_rack_med_1_trunk"],
                "branch": self.canon_target_parts["syn_rack_med_1_branch"],
            }

    def set_demo_info(
        self,
        pc_master_dict,
        demo_idx: int = 0,
        calculate_cost: bool = False,
        show: bool = True,
    ):
        """Process a demonstration."""

        # Get a single demonstration.
        print(demo_idx)
        source_pcd = pc_master_dict["child"]["demo_start_pcds"][demo_idx]
        source_pcd_parts = pc_master_dict["child"]["demo_start_part_pcds"][demo_idx]
        source_start_part_poses = pc_master_dict["child"]["demo_start_part_poses"][
            demo_idx
        ]
        self.source_start_part_poses = source_start_part_poses

        source_start = np.array(
            pc_master_dict["child"]["demo_start_poses"][demo_idx], dtype=np.float64
        )
        source_final = np.array(
            pc_master_dict["child"]["demo_final_poses"][demo_idx], dtype=np.float64
        )

        source_start_pos, source_start_quat = source_start[:3], source_start[3:]
        source_final_pos, source_final_quat = source_final[:3], source_final[3:]
        source_start_trans = utils.pos_quat_to_transform(
            source_start_pos, source_start_quat
        )
        source_final_trans = utils.pos_quat_to_transform(
            source_final_pos, source_final_quat
        )

        target_start = np.array(
            pc_master_dict["parent"]["demo_start_poses"][demo_idx], dtype=np.float64
        )
        target_final = np.array(
            pc_master_dict["parent"]["demo_final_poses"][demo_idx], dtype=np.float64
        )

        target_start_pos, target_start_quat = target_start[:3], target_start[3:]
        target_final_pos, target_final_quat = target_final[:3], target_final[3:]

        target_final_trans = utils.pos_quat_to_transform(
            target_final_pos, target_final_quat
        )

        source_start_to_final = source_final_trans @ np.linalg.inv(source_start_trans)

        part_start_to_finals = {}

        target_pcd = pc_master_dict["parent"]["demo_start_pcds"][demo_idx]
        target_pcd_parts = pc_master_dict["parent"]["demo_start_part_pcds"][demo_idx]
        target_start_part_poses = pc_master_dict["parent"]["demo_start_part_poses"][
            demo_idx
        ]
        self.target_start_part_poses = target_start_part_poses
        print(source_pcd_parts.keys())

        for part in self.source_part_names:
            if (
                self.pcd_subsample_points is not None
                and len(source_pcd_parts[part]) > self.pcd_subsample_points
            ):
                source_pcd_parts[part], _ = utils.farthest_point_sample(
                    source_pcd_parts[part], self.pcd_subsample_points
                )
        # print(self.target_part_names)

        for part in self.target_part_names:
            if (
                self.pcd_subsample_points is not None
                and len(target_pcd_parts[part]) > self.pcd_subsample_points
            ):
                target_pcd_parts[part], _ = utils.farthest_point_sample(
                    target_pcd_parts[part], self.pcd_subsample_points
                )

        # Perception.
        inference_kwargs = {
            "train_latents": True,  # not self.ablate_no_warp,
            "train_scales": True,  # not self.ablate_no_scale,
            "train_poses": True,  # not self.ablate_no_pose_training
        }

        param_1 = cp.deepcopy(PARAM_1)
        if self.ablate_no_size_reg:
            param_1["object_size_reg"] = 0.0

        source_parts = {}
        source_params = {}
        target_parts = {}
        target_params = {}

        if len(self.source_part_names) > 1:
            canon_source_part_labels = get_canon_labels(None,
                self.canon_source_parts, self.source_part_names
            )
            source_part_labels = get_part_labels(
                [{part_1: source_pcd_parts[part_1], part_2: source_pcd_parts[part_2]} for part_1, part_2 in itertools.combinations(self.source_part_names, r=2) if part_1 != part_2], 
                #{part: source_pcd_parts[part] for part in self.source_part_names},
                #part_names=self.source_part_names,
            )

        if len(self.target_part_names) > 1:
            canon_target_part_labels = get_canon_labels(None, 
                self.canon_target_parts, self.target_part_names
            )
            target_part_labels = get_part_labels(
                [{part_1: target_pcd_parts[part_1], part_2: target_pcd_parts[part_2]} for part_1, part_2 in itertools.combinations(self.target_part_names, r=2) if part_1 != part_2], 
                #part_names=self.target_part_names,
            )

        # fig = viz_utils.show_pcds_plotly(
        #     {
        #         "source_cup": source_pcd_parts["cup"],
        #         "canon_cup": self.canon_source_parts["cup"].canonical_pcl,
        #     }
        # )
        # fig.show()

        # Source part warping
        for part in self.source_part_names:
            n_angles = 8
            if len(self.source_part_names) > 1:
                target, target_labels, source, canon_part_labels = (
                    source_pcd_parts[part],
                    source_part_labels[part],
                    self.canon_source_parts[part],
                    canon_source_part_labels[part],
                )

            canonical_means = np.mean(
                np.unique(
                    utils.trunc(self.canon_source_parts[part].canonical_pcl), axis=0
                ),
                axis=0,
            )
            centered_canonical = (
                self.canon_source_parts[part].canonical_pcl - canonical_means
            )
            orig_canonical = cp.deepcopy(self.canon_source_parts[part].canonical_pcl)
            self.canon_source_parts[part].canonical_pcl = centered_canonical

            cost_function = (
                lambda source, target, canon_part_labels: mask_and_cost_batch_pt(
                    target,
                    target_labels,
                    source,
                    canon_part_labels,
                )
            )
            # fig = viz_utils.show_pcds_plotly(
            #     {
            #         "source": source.canonical_pcl,
            #         "target": target,
            #         "source_labels_0": source.canonical_pcl[canon_part_labels == 0],
            #         "source_labels_1": source.canonical_pcl[canon_part_labels == 1],
            #         "target_labels_0": target[target_labels == 0],
            #         "target_labels_1": target[target_labels == 1],
            #     }
            # )
            # fig.show()

            if len(self.source_part_names) > 1:
                warp = ObjectWarpingSE3Batch(
                    self.canon_source_parts[part],
                    source_pcd_parts[part],
                    self.device,
                    canon_labels=canon_part_labels,
                    cost_function=cost_function,
                    **cp.deepcopy(PARAM_1),
                    init_scale=1,
                )
            else:
                warp = ObjectWarpingSE3Batch(
                    self.canon_source_parts[part],
                    source_pcd_parts[part],
                    self.device,
                    **cp.deepcopy(PARAM_1),
                )

            # warp = ObjectWarpingSE2Batch(
            #     self.canon_source_parts[part],
            #     source_pcd_parts[part],
            #     self.device,
            #     canon_labels=canon_part_labels,
            #     cost_function=cost_function,
            #     **cp.deepcopy(PARAM_1),
            #     init_scale=1,
            # )

            # source_parts[part], _, source_params[part] = warp_to_pcd_se2(
            #     warp, n_angles, n_batches=12, inference_kwargs=inference_kwargs
            # )

            source_parts[part], _, source_params[part] = warp_to_pcd_se3_hemisphere(
                warp, n_angles, n_batches=12, inference_kwargs=inference_kwargs
            )

            self.canon_source_parts[part].canonical_pcl = orig_canonical
            final_transform = utils.pos_quat_to_transform(
                source_params[part].position, source_params[part].quat
            ) @ utils.pos_quat_to_transform(-canonical_means, (0, 0, 0, 1))
            full_pos, full_quat = utils.transform_to_pos_quat(final_transform)
            source_params[part].position, source_params[part].quat = full_pos, full_quat

        # Target part warping
        for part in self.target_part_names:
            n_angles = 8
            target, target_labels, source, canon_part_labels = (
                target_pcd_parts[part],
                target_part_labels[part],
                self.canon_target_parts[part],
                canon_target_part_labels[part],
            )

            canonical_means = np.mean(
                np.unique(
                    utils.trunc(self.canon_target_parts[part].canonical_pcl), axis=0
                ),
                axis=0,
            )
            centered_canonical = (
                self.canon_target_parts[part].canonical_pcl - canonical_means
            )
            orig_canonical = cp.deepcopy(self.canon_target_parts[part].canonical_pcl)
            self.canon_target_parts[part].canonical_pcl = centered_canonical

            # fig = viz_utils.show_pcds_plotly(
            #     {
            #         "source": source.canonical_pcl,
            #         "target": target,
            #         "source_labels_0": source.canonical_pcl[canon_part_labels == 0],
            #         "source_labels_1": source.canonical_pcl[canon_part_labels == 1],
            #         "target_labels_0": target[target_labels == 0],
            #         "target_labels_1": target[target_labels == 1],
            #     }
            # )
            # fig.show()

            # warp = ObjectWarpingSE2Batch(#ObjectWarpingSE3Batch(
            #     self.canon_target_parts[part],
            #     target_pcd_parts[part],
            #     self.device,
            #     **cp.deepcopy(PARAM_1),
            # )
            # target_parts[part], _, target_params[part] = warp_to_pcd_se2(
            #     warp, n_angles, n_batches=12, inference_kwargs=inference_kwargs
            # )

            if len(self.target_part_names) > 1:
                warp = ObjectWarpingSE2Batch(
                    self.canon_target_parts[part],
                    target_pcd_parts[part],
                    self.device,
                    canon_labels=canon_part_labels,
                    cost_function=cost_function,
                    **cp.deepcopy(PARAM_1),
                    init_scale=1,
                )
            else:
                warp = ObjectWarpingSE2Batch(
                    self.canon_target_parts[part],
                    target_pcd_parts[part],
                    self.device,
                    **cp.deepcopy(PARAM_1),
                    init_scale=1,
                )

            # warp = ObjectWarpingSE3Batch(#ObjectWarpingSE3Batch(
            #     self.canon_target_parts[part],
            #     target_pcd_parts[part],
            #     self.device,
            #     **cp.deepcopy(PARAM_1),
            # )
            target_parts[part], _, target_params[part] = warp_to_pcd_se2(
                warp, n_angles, n_batches=12, inference_kwargs=inference_kwargs
            )

            self.canon_target_parts[part].canonical_pcl = orig_canonical
            final_transform = utils.pos_quat_to_transform(
                target_params[part].position, target_params[part].quat
            ) @ utils.pos_quat_to_transform(-canonical_means, (0, 0, 0, 1))
            full_pos, full_quat = utils.transform_to_pos_quat(final_transform)
            target_params[part].position, target_params[part].quat = full_pos, full_quat

            # fig = viz_utils.show_pcds_plotly(
            #     {
            #         "source": source.canonical_pcl,
            #         "target": target,
            #         "source_labels_0": self.canon_target_parts[part].to_transformed_pcd(
            #             target_params[part]
            #         )[canon_part_labels == 0],
            #         "source_labels_1": self.canon_target_parts[part].to_transformed_pcd(
            #             target_params[part]
            #         )[canon_part_labels == 1],
            #         "target_labels_0": target[target_labels == 0],
            #         "target_labels_1": target[target_labels == 1],
            #     }
            # )
            # fig.show()

        # viz_utils.show_pcds_plotly({'bowl': source_parts['whole_bowl'], 'cup': target_parts['cup'], 'handle':target_parts['handle'],
        #                             'real_bowl': source_pcd_parts['whole_bowl'], 'real_cup': target_pcd_parts['cup'], 'real_handle': target_pcd_parts['handle']})

        if show:
            print()
            # fig = viz_utils.show_pcds_plotly(
            #     {
            #         "pcd": source_pcd,
            #         "target_cup": source_pcd_parts["cup"],
            #         "target_handle": source_pcd_parts["handle"],
            #         "cup": source_parts["cup"],
            #         "handle": source_parts["handle"],
            #         "target_trunk": target_pcd_parts["trunk"],
            #         "target_branch": target_pcd_parts["branch"],
            #         "trunk": target_parts["trunk"],
            #         "branch": target_parts["branch"],
            #     },
            #     center=False,
            # )
            # fig.show()

        warped_source_part_pcds = {}
        warped_target_part_pcds = {}
        warped_source_meshes = {}
        warped_target_meshes = {}
        for part in self.source_part_names:
            trans = utils.pos_quat_to_transform(
                source_params[part].position, source_params[part].quat
            )
            trans = source_start_to_final @ trans
            pos, quat = utils.transform_to_pos_quat(trans)
            source_params[part].position = pos
            source_params[part].quat = quat

            warped_source_part_pcds[part] = self.canon_source_parts[
                part
            ].to_transformed_pcd(source_params[part])
            warped_source_meshes[part] = self.canon_source_parts[
                part
            ].to_transformed_mesh(source_params[part])

        for part in self.target_part_names:
            trans = utils.pos_quat_to_transform(
                target_params[part].position, target_params[part].quat
            )

            warped_target_part_pcds[part] = self.canon_target_parts[
                part
            ].to_transformed_pcd(target_params[part])
            warped_target_meshes[part] = self.canon_target_parts[
                part
            ].to_transformed_mesh(target_params[part])

        # Move object to final pose.
        warped_source_meshes = {}
        warped_source_part_pcds = {}

        warped_target_meshes = {}
        warped_target_part_pcds = {}

        for part in self.source_part_names:
            warped_source_part_pcds[part] = self.canon_source_parts[
                part
            ].to_transformed_pcd(source_params[part])
            warped_source_meshes[part] = self.canon_source_parts[
                part
            ].to_transformed_mesh(source_params[part])

        for part in self.target_part_names:
            warped_target_part_pcds[part] = self.canon_target_parts[
                part
            ].to_transformed_pcd(target_params[part])
            warped_target_meshes[part] = self.canon_target_parts[
                part
            ].to_transformed_mesh(target_params[part])

        # viz_utils.show_pcds_plotly({'real_demo_bowl': utils.transform_pcd(source_pcd_parts['whole_bowl'], source_start_to_final), 'real_cup': target_pcd_parts['cup'], 'real_handle': target_pcd_parts['handle']})

        mesh = trimesh.util.concatenate(
            [warped_source_meshes[part] for part in self.source_part_names]
        )
        mesh.export(f"tmp_source.obj")
        utils.convex_decomposition(mesh, f"tmp_source_cd.obj")

        mesh = trimesh.util.concatenate(
            [warped_target_meshes[part] for part in self.target_part_names]
        )
        mesh.export("tmp_target.stl")
        utils.convex_decomposition(mesh, "tmp_target.obj")

        # Add predicted meshes to pybullet.
        source_pb = pb.loadURDF("tmp_source.urdf", useFixedBase=True)
        pb.resetBasePositionAndOrientation(
            source_pb, source_final_pos, source_final_quat
        )

        target_pb = pb.loadURDF("tmp_target.urdf", useFixedBase=True)
        pb.resetBasePositionAndOrientation(
            target_pb, target_final_pos, target_final_quat
        )

        source_pcd_complete = np.concatenate(
            [
                self.canon_source_parts[part].to_transformed_pcd(source_params[part])
                for part in self.source_part_names
            ]
        )
        target_pcd_complete = np.concatenate(
            [
                self.canon_target_parts[part].to_pcd(target_params[part])
                for part in self.target_part_names
            ]
        )

        # Save nearby points.
        (
            self.knns,
            self.deltas,
            self.target_indices,
        ) = demo.save_place_nearby_points_by_parts_v2(
            self.source_part_names,
            self.canon_source_parts,
            source_params,
            self.target_part_names,
            self.canon_target_parts,
            target_params,
            self.nearby_points_delta,
        )

        targets_source = {part: {} for part in self.source_part_names}
        targets_target = {part: {} for part in self.source_part_names}
        for part in self.source_part_names:
            for target_part in self.target_part_names:
                if self.knns[part][target_part] is None:
                    continue
                anchors = self.canon_source_parts[part].to_pcd(source_params[part])[
                    self.knns[part][target_part]
                ]
                targets_source[part][target_part] = np.mean(
                    anchors + self.deltas[part][target_part], axis=1
                )
                targets_target[part][target_part] = self.canon_target_parts[
                    target_part
                ].to_pcd(target_params[target_part])[
                    self.target_indices[part][target_part]
                ]  # self.canon_target.to_pcd(target_param)[self.target_indices[part]]

        # cup_transform = utils.pos_quat_to_transform(source_params['cup'].position, source_params['cup'].quat)
        # handle_transform = utils.pos_quat_to_transform(source_params['handle'].position, source_params['handle'].quat)

        # viz_utils.show_pcds_plotly({'cup': source_pcd_parts['cup'],
        #                             'cup_targets': utils.transform_pcd(targets_source['cup']['branch'], cup_transform),
        #                             'handle': source_pcd_parts['handle'],
        #                             'handle_targets': utils.transform_pcd(targets_source['handle']['branch'], handle_transform),
        #                             'tree':target_pcd_complete,
        #                             'tree targets cup': targets_target['cup']['branch'],
        #                             'tree_targets_handle': targets_target['handle']['branch']})

        # Remove predicted meshes from pybullet.
        pb.removeBody(source_pb)
        pb.removeBody(target_pb)

        # return 0, [("cup", "branch"), ("handle", "branch")]
        #return 0, [("whole_bowl", "cup")]
        if calculate_cost:
            print("CALCULATING COST")

            # Make a prediction based on the training sample and calculate the distance between it and the ground-truth.

            possible_pairs = []
            for source_part in self.source_part_names:
                part_pairs = []
                for target_part in self.target_part_names:
                    if self.knns[part][target_part] is None:
                        continue
                    part_pairs.append((source_part, target_part))
                possible_pairs.append(part_pairs)

            possible_constraint_programs = list(itertools.product(*possible_pairs))

            costs = []
            for pair in possible_constraint_programs:
                print(pair)
                trans_predicted = self.infer_relpose(
                    source_pcd_parts, target_pcd_parts, pair, se3=True
                )
                cost = utils.pose_distance(trans_predicted, source_start_to_final)
                costs.append(cost)

            min_cost = np.min(np.array(costs))
            min_pair = possible_constraint_programs[np.argmin(np.array(costs))]

        return min_cost, min_pair

    def infer_relpose(
        self,
        source_pcds,
        target_pcds,
        selected_pairs,
        se3: bool = False,
        show: bool = True,
        experiment_id=None,
        knn_pkl=None,
        child_params=None,
        parent_params=None,
        return_part_transforms = False,
        return_constraint_pcl = False
    ):
        if knn_pkl is not None:
            demo_dict = pickle.load(open(knn_pkl, "rb"))
            self.knns = demo_dict["knns"]
            self.deltas = demo_dict["deltas"]
            self.target_indices = demo_dict["target_indices"]

        for part in self.source_part_names:
            """Make prediction about the final pose of the source object."""
            if (
                self.pcd_subsample_points is not None
                and len(source_pcds[part]) > self.pcd_subsample_points
            ):
                source_pcds[part], _ = utils.farthest_point_sample(
                    source_pcds[part], self.pcd_subsample_points
                )
        for part in self.target_part_names:
            """Make prediction about the final pose of the source object."""
            if (
                self.pcd_subsample_points is not None
                and len(target_pcds[part]) > self.pcd_subsample_points
            ):
                target_pcds[part], _ = utils.farthest_point_sample(
                    target_pcds[part], self.pcd_subsample_points
                )
            print(f"{part} pose: {np.mean(target_pcds[part], axis=0)}")
        # input("continue relpose?")

        inference_kwargs = {
            "train_latents": True,  # not self.ablate_no_warp,
            "train_scales": True,  # not self.ablate_no_scale,
            "train_poses": True,  # not self.ablate_no_pose_training
        }

        param_1 = cp.deepcopy(PARAM_1)
        alignment_param = cp.deepcopy(ALIGNMENT_PARAM)
        if self.ablate_no_size_reg:
            param_1["object_size_reg"] = 0.0

        source_parts_complete = {}
        source_params = {}
        base_handle_tf = None

        canon_source_part_labels = {}
        source_part_labels = {}

        canon_target_part_labels = {}
        target_part_labels = {}

        if len(self.source_part_names) > 1:
            canon_source_part_labels['relational'] = get_canon_labels(None, 
                self.canon_source_parts, self.source_part_names
            )
            source_part_labels['relational'] = get_part_labels(
                [{part_1: source_pcds[part_1], part_2: source_pcds[part_2]} for part_1, part_2 in itertools.combinations(self.source_part_names, r=2) if part_1 != part_2], 
                #{part: source_pcds[part] for part in self.source_part_names},
                #part_names=self.source_part_names,
            )

        canon_source_part_labels['variational'] = get_z_descriptors({part: self.canon_source_parts[part].canonical_pcl for part in self.source_part_names},
                                                                           self.source_part_names)
        source_part_labels['variational'] = get_z_descriptors(source_pcds, self.source_part_names)

        if len(self.target_part_names) > 1:
            canon_target_part_labels['relational'] = get_canon_labels(None, 
                self.canon_target_parts, self.target_part_names
            )
            target_part_labels['relational'] = get_part_labels(
                [{part_1: target_pcds[part_1], part_2:target_pcds[part_2]} for part_1, part_2 in itertools.combinations(self.target_part_names, r=2) if part_1 != part_2], 
                #{part: target_pcds[part] for part in self.target_part_names},
                #part_names=self.target_part_names,
            )
        canon_target_part_labels['variational'] = get_z_descriptors({part: self.canon_target_parts[part].canonical_pcl for part in self.target_part_names}, 
                                                                           self.target_part_names)
        target_part_labels['variational'] = get_z_descriptors(target_pcds, self.target_part_names)
        
        if child_params is None:
            for part in self.source_part_names:
                n_angles = 15
                if len(self.source_part_names) > 1:
                    target, target_labels, source, canon_part_labels = (
                        source_pcds[part],
                        {'variational': source_part_labels['variational'][part], 
                        'relational':  source_part_labels['relational'][part]},
                        self.canon_source_parts[part],
                        {'variational': canon_source_part_labels['variational'][part], 
                        'relational':canon_source_part_labels['relational'][part]},
                    )

                cost_function = (
                    lambda source, target, canon_part_labels, latent_param, scale_param, initial_latents: mask_and_cost_batch_pt(
                        target,
                        target_labels['relational'],
                        source,
                        canon_part_labels['relational'],
                    ) + mask_and_cost_batch_pt(
                        target,
                        target_labels['variational'],
                        source,
                        canon_part_labels['variational'],
                    ) #+ torch.norm(latent_param - initial_latents) * shape_weight
                )

                # cost_function = (
                #     lambda source, target, canon_part_labels: mask_and_cost_batch_pt(
                #         target,
                #         target_labels,
                #         source,
                #         canon_part_labels,
                #     )
                # )
                # fig = viz_utils.show_pcds_plotly({'source': source.canonical_pcl, 'target': target,
                # 'source_labels_0': source.canonical_pcl[canon_part_labels == 0],
                # 'source_labels_1': source.canonical_pcl[canon_part_labels == 1],
                # 'target_labels_0': target[target_labels == 0],
                # 'target_labels_1': target[target_labels == 1]})
                # fig.show()
                if len(self.source_part_names) > 1:
                    warp = ObjectWarpingSE3Batch(
                        self.canon_source_parts[part],
                        source_pcds[part],
                        self.device,
                        canon_labels=canon_part_labels,
                        cost_function=cost_function,
                        **cp.deepcopy(PARAM_1),
                        init_scale=1,
                    )
                else:
                    warp = ObjectWarpingSE3Batch(
                        self.canon_source_parts[part],
                        source_pcds[part],
                        self.device,
                        **param_1,
                        init_scale=self.canon_source_scale,
                    )

                # warp = ObjectWarpingSE2Batch(
                #     self.canon_source_parts[part],
                #     source_pcds[part],
                #     self.device,
                #     canon_labels=canon_part_labels,
                #     cost_function=cost_function,
                #     **cp.deepcopy(PARAM_1),
                #     init_scale=1,
                # )
                # source_parts_complete[part], _, source_params[part] = (
                #     warp_to_pcd_se2(
                #         warp,
                #         n_angles=n_angles,
                #         n_batches=12,
                #         inference_kwargs=inference_kwargs,
                #     )
                # )

                # warp = ObjectWarpingSE3Batch(
                #     self.canon_source_parts[part],
                #     source_pcds[part],
                #     self.device,
                #     **param_1,
                #     init_scale=self.canon_source_scale,
                # )

                (
                    source_parts_complete[part],
                    _,
                    source_params[part],
                ) = warp_to_pcd_se3_hemisphere(
                    warp,
                    n_angles=n_angles,
                    n_batches=12,
                    inference_kwargs=inference_kwargs,
                )
        else:
            source_params = child_params
            source_parts_complete = {part: self.canon_source_parts[part].to_transformed_pcd(source_params[part]) for part in self.source_part_names}

        # viz_utils.show_pcds_plotly(
        #     {
        #         "target handle": source_pcds["handle"],
        #         "reconstr handle": self.canon_source_parts['handle'].to_transformed_pcd(source_params['handle']),
        #         "target_cup": source_pcds["cup"],
        #         "reconstr cup": self.canon_source_parts['cup'].to_transformed_pcd(source_params['cup']),
        #     }
        # )

        target_parts_complete = {}
        target_params = {}
        if parent_params is None:
            for part in self.target_part_names:
                n_angles = 8

                target, target_labels, source, canon_part_labels = (
                    target_pcds[part],
                    {'variational': target_part_labels['variational'][part], 
                    'relational':  target_part_labels['relational'][part]},
                    self.canon_target_parts[part],
                    {'variational': canon_target_part_labels['variational'][part], 
                    'relational':canon_target_part_labels['relational'][part]},
                )

                cost_function = (
                    lambda source, target, canon_part_labels, latent_param, scale_param, initial_latents: mask_and_cost_batch_pt(
                        target,
                        target_labels['relational'],
                        source,
                        canon_part_labels['relational'],
                    ) + mask_and_cost_batch_pt(
                        target,
                        target_labels['variational'],
                        source,
                        canon_part_labels['variational'],
                    )
                )
                # fig = viz_utils.show_pcds_plotly({'source': source.canonical_pcl, 'target': target,
                # 'source_labels_0': source.canonical_pcl[canon_part_labels == 0],
                # 'source_labels_1': source.canonical_pcl[canon_part_labels == 1],
                # 'target_labels_0': target[target_labels == 0],
                # 'target_labels_1': target[target_labels == 1]})
                # fig.show()

                if len(self.target_part_names) > 1:
                    warp = ObjectWarpingSE2Batch(
                        self.canon_target_parts[part],
                        target_pcds[part],
                        self.device,
                        canon_labels=canon_part_labels,
                        cost_function=cost_function,
                        **cp.deepcopy(PARAM_1),
                        init_scale=1,
                    )
                else:
                    warp = ObjectWarpingSE2Batch(
                        self.canon_target_parts[part],
                        target_pcds[part],
                        self.device,
                        **cp.deepcopy(PARAM_1),
                        init_scale=1,
                    )

                # warp = ObjectWarpingSE3Batch(#ObjectWarpingSE3Batch(
                #         self.canon_target_parts[part],
                #         target_pcds[part],
                #         self.device,
                #         **param_1,
                #         init_scale=self.canon_target_scale,
                #     )

                target_parts_complete[part], _, target_params[part] = warp_to_pcd_se2(
                    warp,
                    n_angles=n_angles,
                    n_batches=12,
                    inference_kwargs=inference_kwargs,
                )
        else:
            target_params = parent_params
            target_parts_complete = {part: self.canon_target_parts[part].to_transformed_pcd(target_params[part]) for part in self.target_part_names}

            # warp = ObjectWarpingSE2Batch(#ObjectWarpingSE3Batch(
            #         self.canon_target_parts[part],
            #         target_pcds[part],
            #         self.device,
            #         **param_1,
            #         init_scale=self.canon_target_scale,
            #     )
            # target_parts_complete[part], _, target_params[part] = (
            #     warp_to_pcd_se2(
            #         warp,
            #         n_angles=n_angles,
            #         n_batches=12,
            #         inference_kwargs=inference_kwargs,
            #     )
            # )

        source_pcd = np.concatenate(
            [source_pcds[part] for part in self.source_part_names]
        )
        # source_labels = np.array(
        #     [0 for _ in range(len(source_pcds["cup"]))]
        #     + [1 for _ in range(len(source_pcds["handle"]))]
        # )
        source_labels = np.array(
            list(
                itertools.chain(
                    *[
                        [i for _ in range(len(source_pcds[part]))]
                        for i, part in enumerate(self.source_part_names)
                    ]
                )
            )
        )
        source_pcd_complete = np.concatenate(
            [source_parts_complete[part] for part in self.source_part_names]
        )

        target_pcd = np.concatenate(
            [target_pcds[part] for part in self.target_part_names]
        )
        # target_labels = np.array(
        #     [0 for _ in range(len(target_pcds["trunk"]))]
        #     + [1 for _ in range(len(target_pcds["branch"]))]
        # )
        target_labels = np.array(
            list(
                itertools.chain(
                    *[
                        [i for _ in range(len(target_pcds[part]))]
                        for i, part in enumerate(self.target_part_names)
                    ]
                )
            )
        )
        target_pcd_complete = np.concatenate(
            [target_parts_complete[part] for part in self.target_part_names]
        )

        meshes = {}
        canon_pcds = {}
        transformed_canon_pcds = {}
        for part in self.source_part_names:
            trans = utils.pos_quat_to_transform(
                source_params[part].position, source_params[part].quat
            )

            pos, quat = utils.transform_to_pos_quat(trans)
            source_params[part].position = pos
            source_params[part].quat = quat
            canon_pcds[part] = self.canon_source_parts[part].to_pcd(source_params[part])
            transformed_canon_pcds[part] = self.canon_source_parts[
                part
            ].to_transformed_pcd(source_params[part])
            meshes[part] = self.canon_source_parts[part].to_transformed_mesh(
                source_params[part]
            )

        source_mesh = trimesh.util.concatenate(
            [meshes[part] for part in self.source_part_names]
        )
        canon_source = np.concatenate(
            [transformed_canon_pcds[part] for part in self.source_part_names]
        )

        target_meshes = {}
        target_canon_pcds = {}
        transformed_target_canon_pcds = {}
        for part in self.target_part_names:
            trans = utils.pos_quat_to_transform(
                target_params[part].position, target_params[part].quat
            )

            pos, quat = utils.transform_to_pos_quat(trans)
            target_params[part].position = pos
            target_params[part].quat = quat
            canon_pcds[part] = self.canon_target_parts[part].to_pcd(target_params[part])
            transformed_target_canon_pcds[part] = self.canon_target_parts[
                part
            ].to_transformed_pcd(target_params[part])
            target_meshes[part] = self.canon_target_parts[part].to_transformed_mesh(
                target_params[part]
            )

        target_mesh = trimesh.util.concatenate(
            [target_meshes[part] for part in self.target_part_names]
        )

        trans_cs_to_ct = {}
        target_pcd_complete = np.concatenate(
            [
                self.canon_target_parts[part].to_pcd(target_params[part])
                for part in self.target_part_names
            ]
        )

        targets_source = {part: {} for part in self.source_part_names}
        targets_target = {part: {} for part in self.source_part_names}
        trans_cs_to_ct = {part: {} for part in self.source_part_names}
        for part in self.source_part_names:
            for target_part in self.target_part_names:
                if self.knns[part][target_part] is None:
                    continue
                anchors = self.canon_source_parts[part].to_pcd(source_params[part])[
                    self.knns[part][target_part]
                ]
                targets_source[part][target_part] = np.mean(
                    anchors + self.deltas[part][target_part], axis=1
                )
                targets_target[part][target_part] = self.canon_target_parts[
                    target_part
                ].to_pcd(target_params[target_part])[
                    self.target_indices[part][target_part]
                ]  # target_pcd_complete[self.target_indices[part]]
                trans_cs_to_ct[part][target_part], _, _ = utils.best_fit_transform(
                    np.array(targets_source[part][target_part]).squeeze(),
                    np.array(targets_target[part][target_part]).squeeze(),
                )

        source_pcd_complete = np.concatenate(
            [
                self.canon_source_parts[part].to_transformed_pcd(source_params[part])
                for part in self.source_part_names
            ]
        )
        target_pcd_complete = np.concatenate(
            [
                self.canon_target_parts[part].to_pcd(target_params[part])
                for part in self.target_part_names
            ]
        )

        # cup_transform = utils.pos_quat_to_transform(source_params['cup'].position, source_params['cup'].quat)
        # handle_transform = utils.pos_quat_to_transform(source_params['handle'].position, source_params['handle'].quat)

        # viz_utils.show_pcds_plotly({'cup': transformed_canon_pcds['cup'],
        #                             'cup_targets': utils.transform_pcd(targets_source['cup']['branch'], cup_transform),
        #                             'handle': transformed_canon_pcds['handle'],
        #                             'handle_targets': utils.transform_pcd(targets_source['handle']['branch'], handle_transform),
        #                             'tree':target_pcd_complete,
        #                             'tree targets cup': targets_target['cup']['branch'],
        #                             'tree_targets_handle': targets_target['handle']['branch']})
        target_pcd_complete = np.concatenate(
            [
                self.canon_target_parts[part].to_transformed_pcd(target_params[part])
                for part in self.target_part_names
            ]
        )

        trans_s_to_b = {}
        for part in self.source_part_names:
            trans_s_to_b[part] = utils.pos_quat_to_transform(
                source_params[part].position, source_params[part].quat
            )

        trans_t_to_b = {}
        for part in self.target_part_names:
            trans_t_to_b[part] = utils.pos_quat_to_transform(
                target_params[part].position, target_params[part].quat
            )

        trans_s_to_t = {part: {} for part in self.source_part_names}
        for part in self.source_part_names:
            for target_part in self.target_part_names:
                if self.knns[part][target_part] is None:
                    continue
                trans_s_to_t[part][target_part] = (
                    trans_t_to_b[target_part]
                    @ trans_cs_to_ct[part][target_part]
                    @ np.linalg.inv(trans_s_to_b[part])
                )
        print(trans_s_to_t)
        # show = True
        if show:
            print()
            # fig = viz_utils.show_pcds_plotly(
            #     {
            #         "branch": self.canon_target_parts["branch"].to_transformed_pcd(
            #             target_params["branch"]
            #         ),
            #         "trunk": self.canon_target_parts["trunk"].to_transformed_pcd(
            #             target_params["trunk"]
            #         ),
            #         "cup": self.canon_source_parts["cup"].to_transformed_pcd(
            #             source_params["cup"]
            #         ),
            #         "handle": self.canon_source_parts["handle"].to_transformed_pcd(
            #             source_params["handle"]
            #         ),
            #         "inv_cup": utils.transform_pcd(
            #             source_pcds["cup"], np.linalg.inv(trans_s_to_b["cup"])
            #         ),
            #         "inv_handle": utils.transform_pcd(
            #             source_pcds["handle"], np.linalg.inv(trans_s_to_b["handle"])
            #         ),
            #         "canon_cup": self.canon_source_parts["cup"].canonical_pcl,
            #         "canon_handle": self.canon_source_parts["handle"].canonical_pcl,
            #         "inv_and_cs_cup": utils.transform_pcd(
            #             source_pcds["cup"],
            #             trans_cs_to_ct["cup"]["branch"]
            #             @ np.linalg.inv(trans_s_to_b["cup"]),
            #         ),
            #         "inv_and_cs_handle": utils.transform_pcd(
            #             source_pcds["handle"],
            #             trans_cs_to_ct["handle"]["branch"]
            #             @ np.linalg.inv(trans_s_to_b["handle"]),
            #         ),
            #         "target_branch": target_pcds["branch"],
            #         "target_trunk": target_pcds["trunk"],
            #         "canon_trunk": self.canon_target_parts["trunk"].canonical_pcl,
            #         "canon_branch": self.canon_target_parts["branch"].canonical_pcl,
            #         "source_cup": source_pcds["cup"],
            #         "source_handle": source_pcds["handle"],
            #         "transformed_handle": utils.transform_pcd(
            #             self.canon_source_parts["handle"].to_transformed_pcd(
            #                 source_params["handle"]
            #             ),
            #             trans_s_to_t["handle"]["branch"],
            #         ),
            #         "transformed_cup": utils.transform_pcd(
            #             self.canon_source_parts["cup"].to_transformed_pcd(
            #                 source_params["cup"]
            #             ),
            #             trans_s_to_t["cup"]["branch"],
            #         ),
            #         "cup_branch_targets_source": targets_source["cup"]["branch"],
            #         "handle_branch_targets_source": targets_source["handle"]["branch"],
            #         "cup_branch_targets_target": targets_target["cup"]["branch"],
            #         "handle_branch_targets_target": targets_target["handle"]["branch"],
            #     }
            # )
            # fig.show()

        component_pcls = []
        component_mesh_vertices = []
        component_mesh_faces = []

        for pair in selected_pairs:
            component_pcls.append(
                utils.transform_pcd(
                    transformed_canon_pcds[pair[0]], trans_s_to_t[pair[0]][pair[1]]
                )
            )
            component_mesh_vertices.append(meshes[part].vertices)
            component_mesh_faces.append(
                meshes[part].faces
                + sum([len(pcl) for pcl in component_mesh_vertices[:-1]])
            )
        

        # Building the final alignment constraint pointcloud
        canon_pcl = np.concatenate(component_pcls, axis=0)
        canon_mesh_vertices = np.concatenate(component_mesh_vertices, axis=0)
        canon_mesh_faces = np.concatenate(component_mesh_faces, axis=0)
        center_transform = utils.pos_quat_to_transform(
            np.mean(np.unique(utils.trunc(canon_pcl), axis=0), axis=0),
            np.array([0.0, 0.0, 0.0, 1.0]),
        )

        # viz_utils.show_pcds_plotly({'constraint': canon_pcl,
        #                             # 'tf_cup': utils.transform_pcd(canon_pcl, combined_constraint_transform),
        #                             'targets_target_cup': targets_target['cup']['branch'],
        #                             'targets_target_handle': targets_target['handle']['branch'],
        #                             # 'targets_source': combined_targets,
        #                             'target': target_pcd_complete,
        #     })

        metadata = CanonPartMetadata("none", "none", ["none"], None)
        combined_part = CanonPart(
            canon_pcl,
            canon_mesh_vertices,
            canon_mesh_faces,
            center_transform,
            metadata,
            None,
            None,
        )

        final_inference_kwargs = {
            "train_latents": False,
            "train_scales": False,
            "train_poses": True,
        }

        combined_means = np.mean(
            np.unique(utils.trunc(combined_part.canonical_pcl), axis=0), axis=0
        )

        centered_combined = combined_part.canonical_pcl - combined_means
        real_combined = cp.deepcopy(combined_part.canonical_pcl)

        source_downsampled, source_downsampled_indices = utils.farthest_point_sample(
            source_pcd, 1000
        )
        combined_part.canonical_pcl = centered_combined

        if len(component_pcls) == 1:
            canon_part_labels = np.zeros(component_pcls[0].shape[0])
        else:
            # cup_label = np.zeros(component_pcls[0].shape[0])
            # handle_label = np.ones(component_pcls[1].shape[0])
            # canon_part_labels = np.concatenate([cup_label, handle_label])
            canon_part_labels = np.array(
                list(
                    itertools.chain(
                        *[
                            [i for _ in range(component_pcls[i].shape[0])]
                            for i in range(len(component_pcls))
                        ]
                    )
                )
            )
            
        print(canon_part_labels.shape)
        canon_part_labels = {'relational': canon_part_labels}
        if source_labels is not None:
            canon_part_labels['variational'] = np.concatenate([canon_source_part_labels['variational'][pair[0]] for pair in selected_pairs], -1)
            constraint_variational = np.concatenate([source_part_labels['variational'][pair[0]] for pair in selected_pairs], -1)
            print(constraint_variational.shape)

            if len(self.source_part_names) > 1:
                cost_function = (
                    lambda source, target, canon_part_labels: mask_and_cost_batch_pt(
                        source,
                        canon_part_labels['relational'],
                        target,
                        [source_labels[source_downsampled_indices]],
                    ) + mask_and_cost_batch_pt(
                        source,
                        canon_part_labels['variational'],
                        target,
                        [constraint_variational[0][source_downsampled_indices]],
                    )
                ) 
            else:
                cost_function = (
                    lambda source, target, canon_part_labels: mask_and_cost_batch_pt(
                        source,
                        canon_part_labels['variational'],
                        target,
                        [constraint_variational[0][source_downsampled_indices]],
                    )
                ) 

            combined_warp = ObjectSE3Batch(
                combined_part,
                source_downsampled,
                self.device,
                canon_labels=[canon_part_labels],
                cost_function=cost_function,
                **alignment_param,
                init_scale=1,
            )
        else:
            combined_warp = ObjectSE3Batch(
                combined_part,
                source_downsampled,
                self.device,
                **alignment_param,
                init_scale=1,
            )

        combined_complete, combined_costs, combined_params = warp_to_pcd_se3_hemisphere(
            combined_warp,
            n_angles=15,
            n_batches=12,
            inference_kwargs=final_inference_kwargs,
        )
        final_transform = np.linalg.inv(
            utils.pos_quat_to_transform(combined_params.position, combined_params.quat)
            @ utils.pos_quat_to_transform(-combined_means, (0, 0, 0, 1))
        )

        combined_part.canonical_pcl = real_combined

        if experiment_id is not None:
            best_idx = np.argmin(combined_warp.cost_history[-1])
            print(f"best_idx: {best_idx}")
            print(f"best_cost: {np.min(combined_warp.cost_history[-1])}")
            print(f"best_tranform: {combined_warp.transform_history[0, best_idx]}")
            best_transform_history = []
            best_transforms = []
            step_names = []

            tf2_history = []
            for transform, cost in zip(
                combined_warp.transform_history, combined_warp.cost_history
            ):
                best_trans = transform[best_idx]
                best_transforms.append(
                    utils.transform_pcd(centered_combined, best_trans.astype(float))
                )
                step_names.append(f"COST: {cost[best_idx]}")

            # viz_utils.show_pcds_video_animation_plotly(
            #     moving_pcl_name='Constraint Transformation',
            #     moving_pcl_frames=best_transform_history,
            #     static_pcls={"Start Pointcloud": source_downsampled},
            #     step_names=step_names,
            #     file_name = experiment_id,
            # )

            source_downsampled_means = np.mean(
                np.unique(utils.trunc(source_downsampled), axis=0), axis=0
            )
            source_downsampled = source_downsampled - source_downsampled_means[None]

            slider_fig = viz_utils.show_pcds_slider_animation_plotly(
                moving_pcl_name="Constraint Transformation",
                moving_pcl_frames=best_transforms,
                static_pcls={"Start Pointcloud": source_downsampled},
                step_names=step_names,
            )
            pickle.dump(slider_fig, open(experiment_id + "_slider_fig.pkl", "wb"))
            # slider_fig.show()

            result_fig = viz_utils.show_pcds_plotly(
                {
                    "child pcl": source_pcd,
                    "placement constraint": combined_part.canonical_pcl,
                    "transformed child": utils.transform_pcd(
                        source_pcd, final_transform
                    ),
                    "reverse transformed constraint": utils.transform_pcd(
                        real_combined, np.linalg.inv(final_transform)
                    ),
                    ""
                    #'t2_trans_pcd':utils.transform_pcd(source_pcd, np.linalg.inv(utils.pos_quat_to_transform(combined_params.position, combined_params.quat))),
                    "target": target_pcd_complete,
                }
            )

            pickle.dump(
                result_fig, open(experiment_id + "_final_transform_fig.pkl", "wb")
            )

            # result_fig.show()
            # print(np.mean(utils.transform_pcd(real_combined, np.linalg.inv(final_transform)), axis=0))
            # input("continue?")

        # TODO ask Ondrej what this does
        # Save the mesh and its convex decomposition.
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

        if return_constraint_pcl:
            return final_transform, combined_part.canonical_pcl

        if return_part_transforms:
            return final_transform, trans_s_to_t

        return final_transform


@dataclass
class NDFInterface:
    """Interface between my method and the Relational Neural Descriptor Fields code."""

    canon_source_path: dict = (field(default_factory=lambda: {0: None}),)
    canon_target_path: dict = (field(default_factory=lambda: {0: None}),)
    canon_source_scale: float = 1.0
    canon_target_scale: float = 1.0
    source_whole_name: list = field(default_factory=lambda: ["whole_mug"])
    target_whole_name: list = field(default_factory=lambda: ["mug_tree"])
    pcd_subsample_points: Optional[int] = 1000  # 2000
    nearby_points_delta: float = 0.03
    wiggle: bool = False
    ablate_no_warp: bool = False
    ablate_no_scale: bool = False
    ablate_no_pose_training: bool = False
    ablate_no_size_reg: bool = False

    def __post_init__(self):
        self.canon_source = pickle.load(
            open(self.canon_source_path[self.source_whole_name[0]], "rb")
        )
        self.canon_target = pickle.load(
            open(self.canon_target_path[self.target_whole_name[0]], "rb")
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def set_demo_info(
        self,
        pc_master_dict,
        demo_idx: int = 0,
        calculate_cost: bool = False,
        show: bool = False,
    ):
        """Process a demonstration."""

        # Get a single demonstration.
        source_pcd = pc_master_dict["child"]["demo_start_pcds"][demo_idx]
        source_start = np.array(
            pc_master_dict["child"]["demo_start_poses"][demo_idx], dtype=np.float64
        )
        source_final = np.array(
            pc_master_dict["child"]["demo_final_poses"][demo_idx], dtype=np.float64
        )

        source_start_pos, source_start_quat = source_start[:3], source_start[3:]
        source_final_pos, source_final_quat = source_final[:3], source_final[3:]
        source_start_trans = utils.pos_quat_to_transform(
            source_start_pos, source_start_quat
        )
        self.source_start_trans = source_start_trans
        source_final_trans = utils.pos_quat_to_transform(
            source_final_pos, source_final_quat
        )
        source_start_to_final = source_final_trans @ np.linalg.inv(source_start_trans)

        target_pcd = pc_master_dict["parent"]["demo_start_pcds"][demo_idx]
        if (
            self.pcd_subsample_points is not None
            and len(source_pcd) > self.pcd_subsample_points
        ):
            source_pcd, _ = utils.farthest_point_sample(
                source_pcd, self.pcd_subsample_points
            )
        if (
            self.pcd_subsample_points is not None
            and len(target_pcd) > self.pcd_subsample_points
        ):
            target_pcd, _ = utils.farthest_point_sample(
                target_pcd, self.pcd_subsample_points
            )

        # Perception.
        inference_kwargs = {
            "train_latents": True,  # not self.ablate_no_warp,
            "train_scales": True,  # not self.ablate_no_scale,
            "train_poses": True,  # not self.ablate_no_pose_training
        }

        param_1 = cp.deepcopy(PARAM_1)
        alignment_param = cp.deepcopy(ALIGNMENT_PARAM)
        if self.ablate_no_size_reg:
            param_1["object_size_reg"] = 0.0

        warp = ObjectWarpingSE3Batch(
            self.canon_source,
            source_pcd,
            self.device,
            **param_1,
            init_scale=self.canon_source_scale,
        )
        source_pcd_complete, _, source_param = warp_to_pcd_se3(
            warp, n_angles=8, n_batches=12, inference_kwargs=inference_kwargs
        )

        # warp = ObjectWarpingSE3Batch(
        #     self.canon_source,
        #     source_pcd,
        #     self.device,
        #     **param_1,
        #     init_scale=self.canon_source_scale,
        # )
        # source_pcd_complete, _, source_param = warp_to_pcd_se3(
        #     warp, n_angles=15, n_batches=12, inference_kwargs=inference_kwargs
        # )

        # generate_slider_viz(warp, source_pcd, self.canon_source.canonical_pcl, generate_animation=False)

        warp = ObjectWarpingSE3Batch(
            self.canon_target,
            target_pcd,
            self.device,
            **param_1,
            init_scale=self.canon_target_scale,
        )
        target_pcd_complete, _, target_param = warp_to_pcd_se3(
            warp, n_angles=8, n_batches=12, inference_kwargs=inference_kwargs
        )

        # warp = ObjectWarpingSE3Batch(
        #     self.canon_target,
        #     target_pcd,
        #     self.device,
        #     **param_1,
        #     init_scale=self.canon_target_scale,
        # )
        # target_pcd_complete, _, target_param = warp_to_pcd_se3(
        #     warp, n_angles=15, n_batches=12, inference_kwargs=inference_kwargs
        # )

        # generate_slider_viz(warp, target_pcd, self.canon_target.canonical_pcl, generate_animation=False).show()
        # show = True
        # if show:
        #     fig = viz_utils.show_pcds_plotly(
        #         {
        #             "source_pcd": source_pcd,
        #             "reconstr_source_pcd": source_pcd_complete,
        #             "target_pcd": target_pcd,
        #             "reconstr_target_pcd": target_pcd_complete,
        #         },
        #         center=False,
        #     )
        #     fig.show()
        # input("continue?")

        # Move object to final pose.
        trans = utils.pos_quat_to_transform(source_param.position, source_param.quat)

        trans = source_start_to_final @ trans
        pos, quat = utils.transform_to_pos_quat(trans)
        source_param.position = pos
        source_param.quat = quat

        # Save the mesh and its convex decomposition.
        mesh = self.canon_source.to_mesh(source_param)
        mesh.export("tmp_source.stl")
        utils.convex_decomposition(mesh, "tmp_source.obj")

        mesh = self.canon_target.to_mesh(target_param)
        mesh.export("tmp_target.stl")
        utils.convex_decomposition(mesh, "tmp_target.obj")

        # Add predicted meshes to pybullet.
        source_pb = pb.loadURDF("tmp_source.urdf", useFixedBase=True)
        pb.resetBasePositionAndOrientation(
            source_pb, source_param.position, source_param.quat
        )

        target_pb = pb.loadURDF("tmp_target.urdf", useFixedBase=True)
        pb.resetBasePositionAndOrientation(
            target_pb, target_param.position, target_param.quat
        )

        time.sleep(10.0)

        # Save nearby points.
        self.knns, self.deltas, self.target_indices = demo.save_place_nearby_points_v2(
            source_pb,
            target_pb,
            self.canon_source,
            source_param,
            self.canon_target,
            target_param,
            self.nearby_points_delta,
        )

        anchors = source_pcd_complete[self.knns]
        targets_source = np.mean(anchors + self.deltas, axis=1)

        # Remove predicted meshes from pybullet.
        pb.removeBody(source_pb)
        pb.removeBody(target_pb)

        # trans_predicted = self.infer_relpose(source_pcd, target_pcd, None)
        # viz_utils.show_pcds_plotly(
        #         {
        #             "source_pcd": utils.transform_pcd(source_pcd, trans_predicted),
        #             "source_pcd_gt": utils.transform_pcd(
        #                 source_pcd, source_start_to_final
        #             ),
        #             "target_pcd": target_pcd,
        #         }
        #     ).show()
        # input("continue 2?")

        if calculate_cost:
            # Make a prediction based on the training sample and calculate the distance between it and the ground-truth.
            trans_predicted = self.infer_relpose(source_pcd, target_pcd, None)

            
            # print("STARTING POS")
            # print(source_start_pos)
            # print("FINAL POS")
            # print(source_final_pos)

            return utils.pose_distance(trans_predicted, source_start_to_final), None

    def infer_relpose(
        self,
        source_pcd,
        target_pcd,
        demo_program,
        se3: bool = False,
        show: bool = True,
        experiment_id=None,
        final_alignment=False,
        knn_pkl=None,
    ):
        if knn_pkl is not None:
            demo_dict = pickle.load(open(knn_pkl, "rb"))
            self.knns = demo_dict["knns"]
            self.deltas = demo_dict["deltas"]
            self.target_indices = demo_dict["target_indices"]
        """Make prediction about the final pose of the source object."""
        if (
            self.pcd_subsample_points is not None
            and len(source_pcd) > self.pcd_subsample_points
        ):
            source_pcd, _ = utils.farthest_point_sample(
                source_pcd, self.pcd_subsample_points
            )
        if (
            self.pcd_subsample_points is not None
            and len(target_pcd) > self.pcd_subsample_points
        ):
            target_pcd, _ = utils.farthest_point_sample(
                target_pcd, self.pcd_subsample_points
            )

        inference_kwargs = {
            "train_latents": not self.ablate_no_warp,
            "train_scales": not self.ablate_no_scale,
            "train_poses": not self.ablate_no_pose_training,
        }

        param_1 = cp.deepcopy(PARAM_1)
        if self.ablate_no_size_reg:
            param_1["object_size_reg"] = 0.0

        se3 = True
        if se3:
            warp = ObjectWarpingSE3Batch(
                self.canon_source,
                source_pcd,
                self.device,
                **param_1,
                init_scale=self.canon_source_scale,
            )
            source_pcd_complete, _, source_param = warp_to_pcd_se3(
                warp, n_angles=8, n_batches=12, inference_kwargs=inference_kwargs
            )
        else:
            warp = ObjectWarpingSE2Batch(
                self.canon_source,
                source_pcd,
                self.device,
                **param_1,
                init_scale=self.canon_source_scale,
            )
            source_pcd_complete, _, source_param = warp_to_pcd_se2(
                warp, n_angles=8, n_batches=12, inference_kwargs=inference_kwargs
            )

        warp = ObjectWarpingSE3Batch(
            self.canon_target,
            target_pcd,
            self.device,
            **param_1,
            init_scale=self.canon_target_scale,
        )
        target_pcd_complete, _, target_param = warp_to_pcd_se3(
            warp, n_angles=8, n_batches=12, inference_kwargs=inference_kwargs
        )
        # warp = ObjectWarpingSE3Batch(
        #     self.canon_target,
        #     target_pcd,
        #     self.device,
        #     **param_1,
        #     init_scale=self.canon_target_scale,
        # )
        # target_pcd_complete, _, target_param = warp_to_pcd_se3(
        #     warp, n_angles=15, n_batches=12, inference_kwargs=inference_kwargs
        # )
        # show = True
        if show:
            viz_utils.show_pcds_plotly(
                {"pcd": source_pcd, "warp": source_pcd_complete}, center=False
            )
            fig = viz_utils.show_pcds_plotly(
                {
                    "pcd": target_pcd,
                    "warp": target_pcd_complete,
                },
                center=False,
            )
            # fig.show()

        # input("Continue2?")

        # Save nearby points.
        anchors = self.canon_source.to_pcd(source_param)[self.knns]
        targets_source = np.mean(anchors + self.deltas, axis=1)
        targets_target = self.canon_target.to_pcd(target_param)[self.target_indices]
        # viz_utils.show_pcds_plotly(
        #     {"pcd": self.canon_source.to_pcd(source_param), "anchors": targets_source}
        # )

        # Canonical source obj to canonical target obj.
        trans_cs_to_ct, _, _ = utils.best_fit_transform(targets_source, targets_target)

        trans_s_to_b = utils.pos_quat_to_transform(
            source_param.position, source_param.quat
        )

        trans_t_to_b = utils.pos_quat_to_transform(
            target_param.position, target_param.quat
        )

        # Save the mesh and its convex decomposition.
        mesh = self.canon_source.to_mesh(source_param)
        mesh.export("tmp_source.stl")
        utils.convex_decomposition(mesh, "tmp_source.obj")

        mesh = self.canon_target.to_mesh(target_param)
        mesh.export("tmp_target.stl")
        utils.convex_decomposition(mesh, "tmp_target.obj")

        # Compute relative transform.
        trans_s_to_t = trans_t_to_b @ trans_cs_to_ct @ np.linalg.inv(trans_s_to_b)

        # if final_alignment:
        #     #transform the source pcd by s_to_t
        #     #use that to make the canon constraint

        # TODO move this data saving behind a flag or generally elsewhere
        if experiment_id is not None:
            best_idx = np.argmin(warp.cost_history[-1])
            best_transform_history = []
            best_transforms = []
            step_names = []
            for transform, cost in zip(warp.transform_history, warp.cost_history):
                best_trans = transform[best_idx]
                best_transforms.append(best_trans)
                best_transform_history.append(
                    utils.transform_pcd(self.canon_target.canonical_pcl, best_trans)
                )
                step_names.append(f"COST: {cost[best_idx]}")

            # viz_utils.show_pcds_video_animation_plotly(
            #     moving_pcl_name='Constraint Transformation',
            #     moving_pcl_frames=best_transform_history,
            #     static_pcls={"Start Pointcloud": source_downsampled},
            #     step_names=step_names,
            #     file_name = experiment_id,
            # )
            print("SDLIER FIG TIME")

            # slider_fig = viz_utils.show_pcds_slider_animation_plotly(
            #     moving_pcl_name="Constraint Transformation",
            #     moving_pcl_frames=best_transform_history,
            #     static_pcls={"Start Pointcloud": target_pcd},
            #     step_names=step_names,
            # )
            # slider_fig.show()
            # pickle.dump(slider_fig, open(experiment_id + "_slider_fig.pkl", "wb"))

            result_fig = viz_utils.show_pcds_plotly(
                {
                    "child pcl": source_pcd,
                    "warped_child": source_pcd_complete,
                    "transformed child": utils.transform_pcd(source_pcd, trans_s_to_t),
                    #'t2_trans_pcd':utils.transform_pcd(source_pcd, np.linalg.inv(utils.pos_quat_to_transform(combined_params.position, combined_params.quat))),
                    "warped_parent": self.canon_target.to_transformed_pcd(target_param),
                    "parent pcl": target_pcd,
                }
            )
            # result_fig.show()
            pickle.dump(
                result_fig, open(experiment_id + "_final_transform_fig.pkl", "wb")
            )

        # TODO: Wiggle disabled
        # Add predicted meshes to pybullet.
        # source_pb = pb.loadURDF("tmp_source.urdf", useFixedBase=True)
        # pb.resetBasePositionAndOrientation(
        #     source_pb, *utils.transform_to_pos_quat(trans_s_to_b)
        # )

        # target_pb = pb.loadURDF("tmp_target.urdf", useFixedBase=True)
        # pb.resetBasePositionAndOrientation(
        #     target_pb, *utils.transform_to_pos_quat(trans_t_to_b)
        # )

        # if self.wiggle:
        #     # Wiggle the source object out of collision.
        #     src_pos, src_quat = utils.wiggle(source_pb, target_pb)
        #     trans_s_to_b = utils.pos_quat_to_transform(src_pos, src_quat)

        # # Remove predicted meshes from pybullet.
        # pb.removeBody(source_pb)
        # pb.removeBody(target_pb)

        # Compute relative transform.
        trans_s_to_t = trans_t_to_b @ trans_cs_to_ct @ np.linalg.inv(trans_s_to_b)
        return trans_s_to_t
