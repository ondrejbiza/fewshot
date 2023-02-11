from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
import torch
from torch import nn, optim

from src import utils


class ObjectWarpingSE3Batch:

    def __init__(
        self, canon_obj: utils.CanonObj, pcd: NDArray[np.float32], poses: NDArray[np.float32],
        pca: PCA, device: torch.device, lr: float):
        n_angles = len(poses)

        self.global_means = np.mean(pcd, axis=0)
        pcd = pcd - self.global_means[None]

        self.latent_param = nn.Parameter(torch.zeros((n_angles, pca.n_components), dtype=torch.float32, device=device), requires_grad=True)
        self.center_param = nn.Parameter(torch.zeros((n_angles, 3), dtype=torch.float32, device=device), requires_grad=True)
        self.pose_param = nn.Parameter(torch.tensor(poses, dtype=torch.float32, device=device), requires_grad=True)

        self.means = torch.tensor(pca.mean_, dtype=torch.float32, device=device)
        self.components = torch.tensor(pca.components_, dtype=torch.float32, device=device)
        self.canonical_pcd = torch.tensor(canon_obj.canonical_pcd, dtype=torch.float32, device=device)
        self.pcd = torch.tensor(pcd, dtype=torch.float32, device=device)

        self.optim = optim.Adam([self.latent_param, self.center_param, self.pose_param], lr=lr)

    def inference(
        self, n_steps: int, num_samples: Optional[int], object_size_reg: float
    ) -> Tuple[List[float], List[NDArray[np.float32]], List[utils.ObjParam]]:

        for _ in range(n_steps):

            if num_samples is not None:
                means_, components_, canonical_pcd_ = self.subsample(num_samples)
            else:
                means_, components_, canonical_pcd_ = self.means, self.components, self.canonical_pcd

            self.optim.zero_grad()
            new_pcd = self.create_warped_transformed_pcd(components_, means_, canonical_pcd_)
            cost = cost_batch_pt(self.pcd[None], new_pcd)

            if object_size_reg is not None:
                size = torch.max(torch.sqrt(torch.sum(torch.square(new_pcd), dim=-1)), dim=-1)[0]
                cost = cost + object_size_reg * size

            cost.sum().backward()
            self.optim.step()

        with torch.no_grad():
            # Compute final cost without subsampling.
            new_pcd = self.create_warped_transformed_pcd(self.components, self.means, self.canonical_pcd)
            cost = cost_batch_pt(self.pcd[None], new_pcd)

            # TODO: Should this be included.
            if object_size_reg is not None:
                size = torch.max(torch.sqrt(torch.sum(torch.square(new_pcd), dim=-1)), dim=-1)[0]
                cost = cost + object_size_reg * size

        return self.assemble_output(cost)

    def create_warped_transformed_pcd(
        self, components: torch.Tensor, means: torch.Tensor,
        canonical_pcd: torch.Tensor) -> torch.Tensor:

        rotm = orthogonalize(self.pose_param)
        deltas = (torch.matmul(self.latent_param, components) + means).view((self.latent_param.shape[0], -1, 3))
        new_pcd = canonical_pcd[None] + deltas
        new_pcd = torch.bmm(new_pcd, rotm.permute((0, 2, 1))) + self.center_param[:, None]
        return new_pcd

    def subsample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        indices = np.random.randint(0, self.components.shape[1] // 3, num_samples)
        means_ = self.means.view(-1, 3)[indices].view(-1)
        components_ = self.components.view(self.components.shape[0], -1, 3)[:, indices].view(self.components.shape[0], -1)
        canonical_obj_pt_ = self.canonical_pcd[indices]

        return means_, components_, canonical_obj_pt_

    def assemble_output(self, cost: torch.Tensor) -> Tuple[List[float], List[NDArray[np.float32]], List[utils.ObjParam]]:

        all_costs = []
        all_new_pcds = []
        all_parameters = []

        with torch.no_grad():

            new_pcd = self.create_warped_transformed_pcd(self.components, self.means, self.canonical_pcd)
            rotm = orthogonalize(self.pose_param)

            new_pcd = new_pcd.cpu().numpy()
            new_pcd = new_pcd + self.global_means[None, None]

            for i in range(len(self.latent_param)):
                all_costs.append(cost[i].item())
                all_new_pcds.append(new_pcd[i])

                position = (self.center_param[i].cpu().numpy() + self.global_means).astype(np.float64)
                quat = utils.rotm_to_quat(rotm[i].cpu().numpy())
                latent = self.latent_param[i].cpu().numpy()
                obj_param = utils.ObjParam(position, quat, latent)

                all_parameters.append(obj_param)

        return all_costs, all_new_pcds, all_parameters


def pose_warp_gd_batch(
    pca: PCA, canonical_obj: NDArray, points: NDArray, device: str="cuda:0", n_angles: int=50, n_batches: int=3,
    lr: float=1e-2, n_steps: int=100, object_size_reg: Optional[float]=None, verbose: bool=False,
    num_samples: int=1000,
) -> Tuple[NDArray, float, Tuple[NDArray, NDArray, NDArray]]:
    
    poses = random_ortho_rots_hemisphere(n_angles * n_batches)

    global_means = np.mean(points, axis=0)
    points = points - global_means[None]

    all_new_objects = []
    all_costs = []
    all_parameters = []
    device = torch.device(device)

    for batch_idx in range(n_batches):

        latent = nn.Parameter(torch.zeros((n_angles, pca.n_components), dtype=torch.float32, device=device), requires_grad=True)
        center = nn.Parameter(torch.zeros((n_angles, 3), dtype=torch.float32, device=device), requires_grad=True)
        pose = nn.Parameter(
            torch.tensor(poses[batch_idx * n_angles: (batch_idx + 1) * n_angles], dtype=torch.float32, device=device),
            requires_grad=True
        )
        means = torch.tensor(pca.mean_, dtype=torch.float32, device=device)
        components = torch.tensor(pca.components_, dtype=torch.float32, device=device)
        canonical_obj_pt = torch.tensor(canonical_obj, dtype=torch.float32, device=device)
        points_pt = torch.tensor(points, dtype=torch.float32, device=device)

        opt = optim.Adam([latent, center, pose], lr=lr)

        for i in range(n_steps):

            indices = np.random.randint(0, components.shape[1] // 3, num_samples)
            means_ = means.view(-1, 3)[indices].view(-1)
            components_ = components.view(components.shape[0], -1, 3)[:, indices].view(components.shape[0], -1)
            canonical_obj_pt_ = canonical_obj_pt[indices]

            opt.zero_grad()

            rot = orthogonalize(pose)

            deltas = (torch.matmul(latent, components_) + means_).view((latent.shape[0], -1, 3))
            new_obj = canonical_obj_pt_[None] + deltas
            cost = cost_batch_pt(points_pt[None], torch.bmm(new_obj, rot.permute((0, 2, 1))) + center[:, None])

            if object_size_reg is not None:
                size = torch.max(torch.sqrt(torch.sum(torch.square(new_obj), dim=-1)), dim=-1)[0]
                cost = cost + object_size_reg * size

            if verbose:
                print("cost:", cost)

            cost.sum().backward()
            opt.step()

        with torch.no_grad():

            deltas = (torch.matmul(latent, components) + means).reshape((latent.shape[0], -1, 3))
            rot = orthogonalize(pose)
            new_obj = canonical_obj_pt[None] + deltas
            new_obj = torch.einsum("bnd,bdk->bnk", new_obj, rot.permute((0, 2, 1)))
            new_obj = new_obj + center[:, None]
            new_obj = new_obj.cpu().numpy()
            new_obj = new_obj + global_means[None, None]

            for i in range(len(latent)):
                all_costs.append(cost[i].item())
                all_new_objects.append(new_obj[i])
                all_parameters.append((latent[i].cpu().numpy(), center[i].cpu().numpy() + global_means, rot[i].cpu().numpy()))

        best_idx = np.argmin(all_costs)
        return all_new_objects[best_idx], all_costs[best_idx], all_parameters[best_idx]


def orthogonalize(x: torch.Tensor) -> torch.Tensor:
    """
    Produce an orthogonal frame from two vectors
    x: [B, 2, 3]
    """
    #u = torch.zeros([x.shape[0],3,3], dtype=torch.float32, device=x.device)
    u0 = x[:, 0] / torch.norm(x[:, 0], dim=1)[:, None]
    u1 = x[:, 1] - (torch.sum(u0 * x[:, 1], dim=1))[:, None] * u0
    u1 = u1 / torch.norm(u1, dim=1)[:, None]
    u2 = torch.cross(u0, u1, dim=1)
    return torch.stack([u0, u1, u2], dim=1)


def cost_batch_pt(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # B x N x K
    diff = torch.sqrt(torch.sum(torch.square(source[:, :, None] - target[:, None, :]), dim=3))
    diff_flat = diff.view(diff.shape[0] * diff.shape[1], diff.shape[2])
    c_flat = diff_flat[list(range(len(diff_flat))), torch.argmin(diff_flat, dim=1)]
    c = c_flat.view(diff.shape[0], diff.shape[1])
    return torch.mean(c, dim=1)
