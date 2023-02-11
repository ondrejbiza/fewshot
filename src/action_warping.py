from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from src import utils


class ActionWarping:
    """Action warping.
    """

    def pick_by_single_vertex(
        self, obj_pc_complete: NDArray[np.float32], obj_param: utils.ObjParam,
        demo: utils.PickDemoSingleVertex) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Pick an object by moving the center of a gripper to a single canonical object vertex.
        
        The grasp is possibly more stable but the pose of the gripper does not warp
        to the object geometry. Not good for, e.g., picking up bowls.
        """

        target_pos = obj_pc_complete[demo.target_index]
        target_quat = utils.rotm_to_quat(np.matmul(
            utils.quat_to_rotm(obj_param.quat),
            utils.quat_to_rotm(demo.target_quat)
        ))

        return target_pos, target_quat

    def place_by_virtual_points(
        self, canon_source_obj: utils.CanonObj, source_obj_param: utils.ObjParam,
        canon_target_obj: utils.CanonObj, target_obj_param: utils.ObjParam,
        demo: utils.PlaceDemoVirtualPoints) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:

        source_obj_pcd = canon_source_obj.to_pcd(source_obj_param)
        target_obj_pcd = canon_target_obj.to_pcd(target_obj_param)

        anchors = source_obj_pcd[demo.knns]
        virtual_points = np.mean(anchors + demo.deltas, axis=1)

        targets = target_obj_pcd[demo.target_indices]

        # How to place the source object on the target object.
        # Both objects are in their canonical coordinate frames for convenience.
        trans_source_to_target, _, _ = utils.best_fit_transform(virtual_points, targets)

        trans_target_obj_to_ws = target_obj_param.get_transform()

        # Target position of the object in hand in the workspace.
        trans_source_obj_to_ws = np.matmul(trans_target_obj_to_ws, trans_source_to_target)

        return utils.transform_to_pos_quat(trans_source_obj_to_ws)
