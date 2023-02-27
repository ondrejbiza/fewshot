from typing import Tuple

import numpy as np
from numpy.typing import NDArray
import pybullet as pb

from src import utils


class Simulation:
    """PyBullet simulation.

    Some settings come from https://github.com/caelan/pybullet-planning.
    
    We use the simulation in the real-world experiments to figure out contact points.
    For simulated experiments, we use a different pybullet simulation setup
    from the Neural Descriptor Field papers.
    
    Note that the simulation is populated with objects inferred by our method.
    I.e. we do not require any ground-truth meshes."""

    def __init__(self, use_gui: bool=True):

        if use_gui:
            flag = pb.GUI
        else:
            flag = pb.DIRECT

        self.sim_id = pb.connect(flag)

        if use_gui:
            pb.configureDebugVisualizer(pb.COV_ENABLE_TINY_RENDERER, False, physicsClientId=self.sim_id) # TODO: does this matter?
            pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, True, physicsClientId=self.sim_id)
            pb.configureDebugVisualizer(pb.COV_ENABLE_MOUSE_PICKING, False, physicsClientId=self.sim_id) # mouse moves meshes
            pb.configureDebugVisualizer(pb.COV_ENABLE_KEYBOARD_SHORTCUTS, False, physicsClientId=self.sim_id)

        pb.resetDebugVisualizerCamera(2, 160, -35, np.zeros(3), physicsClientId=self.sim_id)

        self.objects = []
    
    def add_object(self, obj_path: str, obj_pos: NDArray, obj_quat: NDArray, fixed_base: bool=False) -> int:
        obj_id = pb.loadURDF(obj_path, useFixedBase=fixed_base)
        self.set_pose(obj_id, obj_pos, obj_quat)
        self.objects.append(obj_id)
        return obj_id

    def set_pose(self, obj_id: int, obj_pos: NDArray, obj_quat: NDArray):
        utils.pb_set_pose(obj_id, obj_pos, obj_quat, sim_id=self.sim_id)

    def get_pose(self, obj_id: int) -> Tuple[NDArray, NDArray]:
        return utils.pb_get_pose(obj_id, sim_id=self.sim_id)

    def remove_all_objects(self):
        for obj in self.objects:
            pb.removeBody(obj, physicsClientId=self.sim_id)
        self.objects = []

    def wiggle(self, source_obj_id: int, target_obj_id: int) -> Tuple[NDArray, NDArray]:
        return utils.wiggle(source_obj_id, target_obj_id, sim_id=self.sim_id)
