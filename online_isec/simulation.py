from numpy.typing import NDArray
import pybullet as pb
from typing import Tuple

from pybullet_planning.pybullet_tools import utils as pu
import utils


class Simulation:

    def __init__(self):
        pu.connect(use_gui=True, show_sliders=True)
        pu.set_default_camera(distance=2)
        pu.disable_real_time()
        pu.draw_global_system()

        self.objects = []
    
    def add_object(self, obj_path: str, obj_pos: NDArray, obj_quat: NDArray, fixed_base: bool=False) -> int:
        obj_id = pb.loadURDF(obj_path, useFixedBase=fixed_base)
        self.set_pose(obj_id, obj_pos, obj_quat)
        self.objects.append(obj_id)
        return obj_id

    def set_pose(self, obj_id: int, obj_pos: NDArray, obj_quat: NDArray):
        pu.set_pose(obj_id, (obj_pos, obj_quat))

    def get_pose(self, obj_id: int) -> Tuple[NDArray, NDArray]:
        return pu.get_pose(obj_id)

    def remove_all_objects(self):
        for obj in self.objects:
            pb.removeBody(obj)
        self.objects = []

    def wiggle(self, source_obj_id: int, target_obj_id: int) -> Tuple[NDArray, NDArray]:
        return utils.wiggle(source_obj_id, target_obj_id)
