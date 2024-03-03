import time
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from numpy.typing import NDArray
from pycpd.deformable_registration import DeformableRegistration

from torchcpd.deformable_registration import (
    DeformableRegistration as DeformableRegistrationPT,
)


def trimesh_create_verts_surface(mesh: trimesh.Trimesh, num_surface_samples: Optional[int]=1500) -> NDArray[np.float32]:
    surf_points, _ = trimesh.sample.sample_surface_even(
        mesh, num_surface_samples
    )
    return np.array(surf_points, dtype=np.float32)


def cpd_transform(source, target) -> Tuple[NDArray, NDArray]:
    source, target = source.astype(np.float64), target.astype(np.float64)
    reg = DeformableRegistration(X=source, Y=target, tolerance=0.00001)
    t1 = time.time()
    reg.register()
    print("pycpd", time.time() - t1)
    return reg.W, reg.G


def cpd_transform_pt(source, target) -> Tuple[NDArray, NDArray]:
    source, target = source.astype(np.float64), target.astype(np.float64)
    reg = DeformableRegistrationPT(X=source, Y=target, tolerance=0.00001, device="cpu")
    t1 = time.time()
    reg.register()
    print("torchcpd", time.time() - t1)
    return reg.W.cpu().numpy(), reg.G.cpu().numpy()


def main():
    alpha = 0.2
    mug_path1 = "data/mugs/train/0.stl"
    mug_path2 = "data/mugs/train/1.stl"

    mesh1 = trimesh.load(mug_path1)
    mesh2 = trimesh.load(mug_path2)

    points1 = trimesh_create_verts_surface(mesh1)
    points2 = trimesh_create_verts_surface(mesh2)

    w, g = cpd_transform(points2, points1)
    warp = np.dot(g, w)
    warp = np.hstack(warp)
    warp_pycpd = points1 + warp.reshape(-1, 3)

    w, g = cpd_transform_pt(points2, points1)
    warp = np.dot(g, w)
    warp = np.hstack(warp)
    warp_torchcpd = points1 + warp.reshape(-1, 3)

    ax = plt.figure().add_subplot(projection="3d")
    ax.scatter(points1[:, 0], points1[:, 1], points1[:, 2], alpha=alpha, label="source")
    ax.scatter(points2[:, 0], points2[:, 1], points2[:, 2], alpha=alpha, label="target")
    # ax.scatter(warp_pycpd[:, 0], warp_pycpd[:, 1], warp_pycpd[:, 2], alpha=alpha, label="warp pycpd")
    ax.scatter(warp_torchcpd[:, 0], warp_torchcpd[:, 1], warp_torchcpd[:, 2], alpha=alpha, label="warp torchcpd")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
