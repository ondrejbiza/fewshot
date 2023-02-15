from typing import Tuple

import numpy as np
from numpy.typing import NDArray
import pybullet as pb

from src import utils, viz_utils


def get_knn_and_deltas(obj: NDArray, vps: NDArray, k: int=10,
                       show: bool=False) -> Tuple[NDArray, NDArray]:

    if show:
        viz_utils.show_pcds_pyplot({
            "obj": obj,
            "vps": vps
        })

    dists = np.sum(np.square(obj[None] - vps[:, None]), axis=-1)
    knn_list = []
    deltas_list = []

    for i in range(dists.shape[0]):
        # Get K closest points, compute relative vectors.
        knn = np.argpartition(dists[i], k)[:k]
        deltas = vps[i: i + 1] - obj[knn]
        knn_list.append(knn)
        deltas_list.append(deltas)

    knn_list = np.stack(knn_list)
    deltas_list = np.stack(deltas_list)
    return knn_list, deltas_list



def save_place_nearby_points(source: int, target: int, canon_source_obj: utils.CanonObj, 
                             source_obj_param: utils.ObjParam, canon_target_obj: utils.CanonObj,
                             target_obj_param: utils.ObjParam, delta: float) -> Tuple[NDArray[np.int32], NDArray[np.float32], NDArray[np.int32]]:

    pb.performCollisionDetection()
    cols = pb.getClosestPoints(source, target, delta)

    pos_mug = [col[5] for col in cols]
    pos_tree = [col[6] for col in cols]

    assert len(pos_mug) > 0

    pos_mug = np.stack(pos_mug, axis=0).astype(np.float32)
    pos_tree = np.stack(pos_tree, axis=0).astype(np.float32)

    full_source_pcd = canon_source_obj.to_transformed_pcd(source_obj_param)
    full_target_pcd = canon_target_obj.to_transformed_pcd(target_obj_param)

    knns, deltas = get_knn_and_deltas(full_source_pcd, pos_tree)

    dist_2 = np.sqrt(np.sum(np.square(full_target_pcd[:, None] - pos_tree[None]), axis=2))
    i_2 = np.argmin(dist_2, axis=0).transpose()

    return knns, deltas, i_2
