import os
import pybullet as pb
from .robot_base import RobotBase


class UR5_Simple(RobotBase):

    def __init__(self):

        super(UR5_Simple, self).__init__()
        # Setup arm and gripper variables
        self.max_forces = [150, 150, 150, 28, 28, 28, 30, 30]
        self.gripper_close_force = [30] * 2
        self.gripper_open_force = [30] * 2
        self.end_effector_index = 12

        self.home_positions = [0., 0., -2.137, 1.432, -0.915, -1.591, 0.071, 0., 0., 0., 0., 0., 0., 0.]
        self.home_positions_joint = self.home_positions[1:7]
        self.root_dir = "."

        # TODO: this was previously [0, 0.036], we changed it to work better on the real robot
        # TODO: we also changed the finger width from 0.25 to 0.1 in the ur5_simple_gripper urdf file
        self.gripper_joint_limit = [0.01, 0.036]
        self.gripper_joint_names = list()
        self.gripper_joint_indices = list()

    def initialize(self):

        ur5_urdf_filepath = os.path.join(self.root_dir, 'urdf/ur5/ur5_simple_gripper.urdf')
        self.id = pb.loadURDF(ur5_urdf_filepath, [0,0,0], [0,0,0,1])
        # self.is_holding = False
        self.gripper_closed = False
        self.holding_obj = None
        self.num_joints = pb.getNumJoints(self.id)
        [pb.resetJointState(self.id, idx, self.home_positions[idx]) for idx in range(self.num_joints)]

        self.arm_joint_names = list()
        self.arm_joint_indices = list()
        self.gripper_joint_names = list()
        self.gripper_joint_indices = list()
        for i in range (self.num_joints):
            joint_info = pb.getJointInfo(self.id, i)
            if i in range(1, 7):
                self.arm_joint_names.append(str(joint_info[1]))
                self.arm_joint_indices.append(i)
            elif i in range(10, 12):
                self.gripper_joint_names.append(str(joint_info[1]))
                self.gripper_joint_indices.append(i)

    def reset(self):

        self.gripper_closed = False
        self.holding_obj = None
        [pb.resetJointState(self.id, idx, self.home_positions[idx]) for idx in range(self.num_joints)]

    def close_gripper(self, max_it=100):

        p1, p2 = self.get_gripper_joint_position_()
        limit = self.gripper_joint_limit[1]
        self.send_gripper_command_(limit, limit)
        # self._sendGripperCloseCommand()
        self.gripper_closed = True
        it = 0
        while (limit-p1) + (limit-p2) > 0.001:
        # while p1 < 0.036:
            pb.stepSimulation()
            it += 1
            p1_, p2_ = self.get_gripper_joint_position_()
            if it > max_it or (abs(p1-p1_)<0.0001 and abs(p2-p2_)<0.0001):
                mean = (p1+p2)/2 + 0.01
                self.send_gripper_command_(mean, mean)
                return False
            p1 = p1_
            p2 = p2_
        return True

    def check_gripper_closed(self):

        limit = self.gripper_joint_limit[1]
        p1, p2 = self.get_gripper_joint_position_()
        if (limit - p1) + (limit - p2) > 0.001:
            return
        else:
            self.holding_obj = None

    def open_gripper(self):

        p1, p2 = self.get_gripper_joint_position_()
        limit = self.gripper_joint_limit[0]
        self.send_gripper_command_(limit, limit)
        self.gripper_closed = False
        it = 0
        while p1 > 0.0:
            pb.stepSimulation()
            it += 1
            if it > 100:
                return False
            p1, p2 = self.get_gripper_joint_position_()
        return True

    def adjust_gripper_command_(self):

        p1, p2 = self.get_gripper_joint_position_()
        mean = (p1 + p2) / 2 + 0.01
        self.send_gripper_command_(mean, mean)

    def calculate_IK_(self, pos, rot):

        return pb.calculateInverseKinematics(self.id, self.end_effector_index, pos, rot)[:-2]

    def get_gripper_joint_position_(self):

        p1 = pb.getJointState(self.id, self.gripper_joint_indices[0])[0]
        p2 = pb.getJointState(self.id, self.gripper_joint_indices[1])[0]
        return p1, p2

    def send_position_command_(self, commands):

        num_motors = len(self.arm_joint_indices)
        pb.setJointMotorControlArray(self.id, self.arm_joint_indices, pb.POSITION_CONTROL, commands,
                                     [0.]*num_motors, self.max_forces[:-2], [0.02]*num_motors, [1.0]*num_motors)

    def send_gripper_command_(self, target_pos1, target_pos2):

        pb.setJointMotorControlArray(self.id, self.gripper_joint_indices, pb.POSITION_CONTROL,
                                     targetPositions=[target_pos1, target_pos2], forces=self.gripper_open_force,
                                     positionGains=[0.02]*2, velocityGains=[1.0]*2)
