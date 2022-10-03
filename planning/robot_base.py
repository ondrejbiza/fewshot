import os
import copy
import numpy as np
from collections import deque
from abc import abstractmethod
from typing import List
import pybullet as pb
from . import pybullet_util
from . import transformations
from .pybullet_object import PybulletObject


class RobotBase:

    def __init__(self):

        self.root_dir = "."
        self.id = None
        self.num_joints = None
        self.arm_joint_names = list()
        self.arm_joint_indices = list()
        self.home_positions = None
        self.home_positions_joint = None
        self.end_effector_index = None
        self.holding_obj = None
        self.gripper_closed = False
        self.state = {
            'holding_obj': self.holding_obj,
            'gripper_closed': self.gripper_closed
        }

    def save_state(self):

        self.state = {
            'holding_obj': self.holding_obj,
            'gripper_closed': self.gripper_closed
        }

    def restore_state(self):

        self.holding_obj = self.state['holding_obj']
        self.gripper_closed = self.state['gripper_closed']
        if self.gripper_closed:
            self.close_gripper(max_it=0)
        else:
            self.open_gripper()

    def get_picked_obj(self, objects: List[PybulletObject]):

        if not objects:
            return None
        end_pos = self.get_end_effector_position_()
        sorted_obj = sorted(objects, key=lambda o: np.linalg.norm(end_pos-o.get_position()))
        obj_pos = sorted_obj[0].get_position()
        if np.linalg.norm(end_pos[:-1]-obj_pos[:-1]) < 0.05 and np.abs(end_pos[-1]-obj_pos[-1]) < 0.025:
            return sorted_obj[0]

    def pick(self, pos, rot, offset, dynamic=True, objects=None, simulate_grasp=True):
        # Setup pre-grasp pos and default orientation
        self.open_gripper()
        pre_pos = copy.copy(pos)
        pre_pos[2] += offset
        # rot = pb.getQuaternionFromEuler([np.pi/2.,-np.pi,np.pi/2])
        pre_rot = rot

        # Move to pre-grasp pose and then grasp pose
        self.move_to(pre_pos, pre_rot, dynamic)
        if simulate_grasp:
            self.move_to(pos, rot, True)
            # Grasp object and lift up to pre pose
            gripper_fully_closed = self.close_gripper()
            if gripper_fully_closed:
                self.open_gripper()
                self.move_to(pre_pos, pre_rot, dynamic)
            else:
                self.move_to(pre_pos, pre_rot, True)
                self.adjust_gripper_command_()
                for i in range(10):
                    pb.stepSimulation()
                self.holding_obj = self.get_picked_obj(objects)

        else:
            self.move_to(pos, rot, dynamic)
            self.holding_obj = self.get_picked_obj(objects)

        self.move_to_j(self.home_positions_joint, dynamic)
        self.check_gripper_closed()

    def place(self, pos, rot, offset, dynamic=True, simulate_grasp=True):

        # Setup pre-grasp pos and default orientation
        pre_pos = copy.copy(pos)
        pre_pos[2] += offset
        pre_rot = rot

        # Move to pre-grasp pose and then grasp pose
        self.move_to(pre_pos, pre_rot, dynamic)
        if simulate_grasp:
            self.move_to(pos, rot, True)
        else:
            self.move_to(pos, rot, dynamic)

        # Grasp object and lift up to pre pose
        self.open_gripper()
        self.holding_obj = None
        self.move_to(pre_pos, pre_rot, dynamic)
        self.move_to_j(self.home_positions_joint, dynamic)

    def move_to(self, pos, rot, dynamic=True):

        if dynamic or not self.holding_obj:
            self.move_to_cartesian_pose_(pos, rot, dynamic)
        else:
            self.teleport_arm_with_obj_(pos, rot)

    def move_to_j(self, pose, dynamic=True):

        if dynamic or not self.holding_obj:
            self.move_to_joint_pose_(pose, dynamic)
        else:
            self.teleport_arm_with_obj_joint_pose_(pose)

    @abstractmethod
    def open_gripper(self):

        raise NotImplementedError

    @abstractmethod
    def close_gripper(self, max_it=100):

        raise NotImplementedError

    @abstractmethod
    def check_gripper_closed(self):

        raise NotImplementedError

    def move_to_joint_pose_(self, target_pose, dynamic=True, max_it=1000):

        if dynamic:
            self.send_position_command_(target_pose)
            past_joint_pos = deque(maxlen=5)
            joint_state = pb.getJointStates(self.id, self.arm_joint_indices)
            joint_pos = list(zip(*joint_state))[0]
            n_it = 0
            while not np.allclose(joint_pos, target_pose, atol=1e-2) and n_it < max_it:
                pb.stepSimulation()
                n_it += 1
                # Check to see if the arm can't move any close to the desired joint position
                if len(past_joint_pos) == 5 and np.allclose(past_joint_pos[-1], past_joint_pos, atol=1e-3):
                  break
                past_joint_pos.append(joint_pos)
                joint_state = pb.getJointStates(self.id, self.arm_joint_indices)
                joint_pos = list(zip(*joint_state))[0]

        else:
            self.set_joint_poses_(target_pose)

    def move_to_cartesian_pose_(self, pos, rot, dynamic=True):

        close_enough = False
        outer_it = 0
        threshold = 1e-3
        max_outer_it = 10
        max_inner_it = 1000

        while not close_enough and outer_it < max_outer_it:
            ik_solve = self.calculate_IK_(pos, rot)
            self.move_to_joint_pose_(ik_solve, dynamic, max_inner_it)

            ls = pb.getLinkState(self.id, self.end_effector_index)
            new_pos = list(ls[4])
            new_rot = list(ls[5])
            close_enough = np.allclose(np.array(new_pos + new_rot), np.array(list(pos) + list(rot)), atol=threshold)
            outer_it += 1

    @abstractmethod
    def calculate_IK_(self, pos, rot):

        raise NotImplementedError

    def teleport_arm_with_obj_(self, pos, rot):

        if not self.holding_obj:
            self.move_to_cartesian_pose_(pos, rot, False)
            return

        end_pos = self.get_end_effector_position_()
        end_rot = self.get_end_effector_rotation_()
        obj_pos, obj_rot = self.holding_obj.get_pose()
        oTend = pybullet_util.get_matrix(end_pos, end_rot)
        oTobj = pybullet_util.get_matrix(obj_pos, obj_rot)
        endTobj = np.linalg.inv(oTend).dot(oTobj)

        self.move_to_cartesian_pose_(pos, rot, False)
        end_pos_ = self.get_end_effector_position_()
        end_rot_ = self.get_end_effector_rotation_()
        oTend_ = pybullet_util.get_matrix(end_pos_, end_rot_)
        oTobj_ = oTend_.dot(endTobj)
        obj_pos_ = oTobj_[:3, -1]
        obj_rot_ = transformations.quaternion_from_matrix(oTobj_)

        self.holding_obj.reset_pose(obj_pos_, obj_rot_)

    def teleport_arm_with_obj_joint_pose_(self, joint_pose):

        if not self.holding_obj:
            self.move_to_joint_pose_(joint_pose, False)
            return

        end_pos = self.get_end_effector_position_()
        end_rot = self.get_end_effector_rotation_()
        obj_pos, obj_rot = self.holding_obj.get_pose()
        oTend = pybullet_util.get_matrix(end_pos, end_rot)
        oTobj = pybullet_util.get_matrix(obj_pos, obj_rot)
        endTobj = np.linalg.inv(oTend).dot(oTobj)

        self.move_to_joint_pose_(joint_pose, False)
        end_pos_ = self.get_end_effector_position_()
        end_rot_ = self.get_end_effector_rotation_()
        oTend_ = pybullet_util.get_matrix(end_pos_, end_rot_)
        oTobj_ = oTend_.dot(endTobj)
        obj_pos_ = oTobj_[:3, -1]
        obj_rot_ = transformations.quaternion_from_matrix(oTobj_)

        self.holding_obj.reset_pose(obj_pos_, obj_rot_)

    def get_end_effector_position_(self):

        state = pb.getLinkState(self.id, self.end_effector_index)
        return np.array(state[4])

    def get_end_effector_rotation_(self):

        state = pb.getLinkState(self.id, self.end_effector_index)
        return np.array(state[5])

    @abstractmethod
    def get_gripper_joint_position_(self):

        raise NotImplementedError

    @abstractmethod
    def send_position_command_(self, commands):

        raise NotImplementedError

    @abstractmethod
    def adjust_gripper_command_(self):

        raise NotImplementedError

    def set_joint_poses_(self, q_poses):

        for i in range(len(q_poses)):
            motor = self.arm_joint_indices[i]
            pb.resetJointState(self.id, motor, q_poses[i])

        self.send_position_command_(q_poses)
