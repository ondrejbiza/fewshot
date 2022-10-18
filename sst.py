import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import utils
import sst_utils


class SST:

    DEVICE = "cuda:0"
    VIZ = True
    LEARNING_RATE = 5e-3
    NUM_STEPS = 2000
    SCALE_FACTOR = 2

    def __init__(self, canonical_obj_dict, use_rot=False):

        self.device = self.DEVICE
        self.cam_config = utils.RealSenseD415.CONFIG
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])
        self.pix_size = 0.003125
        self.total_steps = 0

        self.canonical_obj_dict = canonical_obj_dict
        self.use_rot = use_rot

    def get_per_object_representation(self, object_obs, action=None):

        centers = []
        angles = []
        latents = []
        reconstructions = []

        if -1 in self.canonical_obj_dict:
            # repeat the first object K - 1 times
            # the Kth object is the canonical obj with the index -1
            # keys = list(sorted(object_obs.keys()))
            # tmp_canonical = {}
            # for key in keys[:-1]:
            #     tmp_canonical[key] = self.canonical_obj_dict[keys[0]]
            # tmp_canonical[keys[-1]] = self.canonical_obj_dict[-1]

            # or the opposite ...
            keys = list(sorted(object_obs.keys()))
            tmp_canonical = {}
            for key in keys[1:]:
                tmp_canonical[key] = self.canonical_obj_dict[keys[0]]
            tmp_canonical[keys[0]] = self.canonical_obj_dict[-1]
        else:
            tmp_canonical = self.canonical_obj_dict

        for key in sorted(object_obs.keys()):

            import time
            start = time.time()

            if self.VIZ:
                tmp_centers, tmp_angles, tmp_latents, tmp_rec, _ = \
                    self.get_center_and_encoding(object_obs[key], tmp_canonical[key], return_obj=True)
                reconstructions.append(tmp_rec)
            else:
                tmp_centers, tmp_angles, tmp_latents, _ = \
                    self.get_center_and_encoding(object_obs[key], tmp_canonical[key], return_obj=False)

            # print(time.time() - start)

            centers.append(tmp_centers)
            angles.append(tmp_angles)
            latents.append(tmp_latents)

        centers = np.stack(centers, axis=0)
        angles = np.stack(angles, axis=0)
        latents = np.stack(latents, axis=0)

        if self.VIZ:
            reconstructions = np.stack(reconstructions, axis=0)
            points = []
            colors = []
            for i, key in enumerate(sorted(object_obs.keys())):
                points.append(object_obs[key])
                colors.append(np.zeros_like(object_obs[key]) + 0.9)
                points.append(reconstructions[i])
                colors.append(np.zeros_like(reconstructions[i]) + 0.0)
                points.append(centers[i: i + 1])
                c = np.zeros_like(centers[i: i + 1])
                c[:, 0] = 1.
                colors.append(c)
                if action is not None:
                    tmp = np.stack([action["pose0"][0], action["pose1"][0]], axis=0)
                    points.append(tmp)
                    c = np.zeros_like(tmp)
                    c[:, 1] = 1.
                    colors.append(c)
            points = np.concatenate(points, axis=0)
            colors = np.concatenate(colors, axis=0)
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([pcd])

        return centers, angles, latents

    def get_segmented_point_cloud(self, obs):

        return utils.reconstruct_segmented_point_cloud(
            obs["color"], obs["depth"], obs["segm"], self.cam_config, self.bounds, self.pix_size
        )

    def get_center_and_encoding(self, object_obs, pca_data, return_obj=False):

        npoints = 2000
        if len(object_obs) > npoints:
            object_obs = utils.farthest_point_sample(object_obs, npoints)[0]

        pca, canonical_obj = pca_data

        object_obs = np.copy(object_obs)
        object_obs_mean = np.mean(object_obs, axis=0)
        object_obs -= object_obs_mean[None]

        device = torch.device(self.DEVICE)
        latent = nn.Parameter(torch.zeros(4, dtype=torch.float32, device=device), requires_grad=True)
        center = nn.Parameter(torch.zeros(3, dtype=torch.float32, device=device), requires_grad=True)
        means = torch.tensor(pca.mean_, dtype=torch.float32, device=device)
        components = torch.tensor(pca.components_, dtype=torch.float32, device=device)
        canonical_obj_pt = torch.tensor(canonical_obj, dtype=torch.float32, device=device)
        points_pt = torch.tensor(object_obs, dtype=torch.float32, device=device)

        params = [latent, center]
        if self.use_rot:
            angle = nn.Parameter(
                torch.tensor([start_pose], dtype=torch.float32, device=device),
                requires_grad=True
            )
            params.append(angle)

        opt = optim.Adam(params, lr=self.LEARNING_RATE)
        costs = []

        for i in range(self.NUM_STEPS):

            opt.zero_grad()

            if self.use_rot:
                new_obj = self.params_to_object_rot(canonical_obj_pt, latent, components, means, center, angle)
            else:
                new_obj = self.params_to_object(canonical_obj_pt, latent, components, means, center)

            cost = sst_utils.cost_pt(points_pt, new_obj)

            cost.backward()
            opt.step()

            costs.append(cost.detach().item())
            # if i % 10 == 0:
            #     utils.show_pc_and_object(points_pt.detach().cpu().numpy(), new_obj.detach().cpu().numpy())

        # plt.plot(costs)
        # plt.yscale("log")
        # plt.show()
        # print("final cost: {:.4f}".format(cost.detach().item()))

        latent_np = latent.detach().cpu().numpy()
        center_np = center.detach().cpu().numpy() + object_obs_mean
        if self.use_rot:
            angle_np = angle.detach().cpu().numpy()
        else:
            angle_np = np.zeros((1,), dtype=np.float32)

        ret = [center_np, angle_np, latent_np]

        if return_obj:

            with torch.no_grad():

                if self.use_rot:
                    new_obj = self.params_to_object_rot(canonical_obj_pt, latent, components, means, center, angle)
                else:
                    new_obj = self.params_to_object(canonical_obj_pt, latent, components, means, center)

                new_obj = new_obj.cpu().numpy()
                new_obj = new_obj + object_obs_mean[None]

            ret.append(new_obj)

        ret.append(cost.item())
        return ret

    def params_to_object(self, canonical_obj_pt, latent, components, means, center):

        new_obj = self.warp(canonical_obj_pt, latent, components, means)
        new_obj = new_obj + center[None]
        return new_obj

    def params_to_object_rot(self, canonical_obj_pt, latent, components, means, center, angle):

        new_obj = self.warp(canonical_obj_pt, latent, components, means)
        rot = utils.yaw_to_rot_pt(angle)
        new_obj = torch.matmul(rot, new_obj.T).T
        new_obj = new_obj + center[None]
        return new_obj

    def warp(self, canonical_obj_pt, latent, components, means):

        deltas = (torch.matmul(latent[None, :], components)[0] + means).reshape((-1, 3))
        new_obj = canonical_obj_pt + deltas / self.SCALE_FACTOR
        return new_obj

    def warp_batch(self, canonical_obj_pt, latent, components, means):

        deltas = (torch.matmul(latent, components) + means[None]).reshape((latent.size(0), -1, 3))
        new_obj = canonical_obj_pt + deltas / self.SCALE_FACTOR
        return new_obj
