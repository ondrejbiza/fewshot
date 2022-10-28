from typing import Tuple, Dict, List, Any, Optional
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pybullet_planning.pybullet_tools.ikfast import ikfast as ikf
from pybullet_planning.pybullet_tools.ikfast import utils as ikfu
from pybullet_planning.pybullet_tools.ikfast.franka_panda.ik import PANDA_INFO, FRANKA_URDF
from pybullet_planning.pybullet_tools import utils as pu
import utils


@dataclass
class Robot:

    robot: int
    info: ikfu.IKFastInfo
    tool_name: str
    finger_names: Tuple[str, ...]
    init_tool_pos = NDArray
    
    real_time: bool = False
    refinement_step: int = 10
    wait_time: float = 0.002

    def __post_init__(self):
        self.tool_link: int = pu.link_from_name(self.robot, self.tool_name)
        self.ik_joints: List[int] = ikf.get_ik_joints(self.robot, self.info, self.tool_link)
        self.fingers: List[int] = [pu.joint_from_name(self.robot, name) for name in self.finger_names]

        self.move_to_default_pose()

    def move_to_default_pose(self):
        pose = pu.get_link_pose(self.robot, self.tool_link)
        pose = (self.init_tool_pos, pose[1])

        conf = next(ikf.either_inverse_kinematics(
            self.robot, self.info, self.tool_link, pose, max_distance=pu.INF,
            max_time=pu.INF, max_attempts=200, use_pybullet=False
        ), None)
        assert conf is not None
        pu.set_joint_positions(self.robot, self.ik_joints, conf)

    def plan_grasp(
        self, pose: Tuple[NDArray, NDArray], delta: float, obstacles: List[int]
    ) -> List[Tuple[float, ...]]:
        """Get joint configuration for hand pose and plan a collision free path."""
        approach_pose = utils.move_hand_back(pose, delta)

        saved_world = pu.WorldSaver()

        conf = next(ikf.either_inverse_kinematics(
            self.robot, self.info, self.tool_link, approach_pose, max_distance=pu.INF,
            max_time=1., max_attempts=pu.INF, use_pybullet=False
        ), None)
        assert conf is not None

        #saved_world.restore()

        path = pu.plan_joint_motion(self.robot, self.ik_joints, conf, obstacles=obstacles)
        assert path is not None

        saved_world.restore()

        return path

    def plan_grasp_naive(
        self, pose: Tuple[NDArray, NDArray], delta: float
    ) -> List[Tuple[float, ...]]:
        """Get joint configuration for hand pose and simply interpolate it."""
        approach_pose = utils.move_hand_back(pose, delta)

        conf2 = next(ikf.either_inverse_kinematics(
            self.robot, self.info, self.tool_link, approach_pose, max_distance=pu.INF,
            max_time=1., max_attempts=pu.INF, use_pybullet=False
        ), None)
        assert conf2 is not None

        conf1 = pu.get_joint_positions(self.robot, self.ik_joints)
        path = [conf1, conf2]

        return path

    def open_hand(self):
        conf1 = pu.get_joint_positions(self.robot, self.fingers)
        conf2 = [pu.get_joint_limits(self.robot, finger)[1] for finger in self.fingers]
        path = [conf1, conf2]
        self.execute_path(path, fingers=True)

    def close_hand(self):
        # TODO: copy https://github.com/ColinKohler/BulletArm/blob/main/bulletarm/pybullet/robots/ur5_simple.py
        conf1 = pu.get_joint_positions(self.robot, self.fingers)
        conf2 = [pu.get_joint_limits(self.robot, finger)[0] for finger in self.fingers]
        path = [conf1, conf2]
        self.execute_path(path, fingers=True)

    def execute_path(self, path: List[Tuple[float, ...]], fingers: bool=False):
        """Refine and execute a motion plan."""
        # TODO: I could instead use pybullet motor control: setJointMotorControl2.
        # https://github.com/ColinKohler/BulletArm/blob/main/bulletarm/pybullet/robots/ur5_simple.py
        # https://github.com/ColinKohler/BulletArm/blob/cb744b4212b9c0049c6801ba82805a0eaa07d8c7/bulletarm/pybullet/robots/robot_base.py#L290
        if fingers:
            joints = self.fingers
        else:
            joints = self.ik_joints

        path = pu.refine_path(self.robot, joints, path, self.refinement_step)

        for conf in path:
            pu.set_joint_positions(self.robot, joints, conf)
            if not self.real_time:
                pu.step_simulation()
            pu.wait_for_duration(self.wait_time)


@dataclass
class Panda(Robot):

    info: ikfu.IKFastInfo = PANDA_INFO
    tool_name: str = "panda_hand"
    finger_names: Tuple[str, ...] = ("panda_finger_joint1", "panda_finger_joint2")
    init_tool_pos = np.array([0., 0.27, 0.59], dtype=np.float32)
