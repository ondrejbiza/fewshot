from numpy.typing import NDArray
import moveit_commander
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
import rospy

import src.real_world.utils as rw_utils
from src.real_world.robotiq_gripper import RobotiqGripper


class MoveItScene:

    def __init__(self):

        self.scene = moveit_commander.PlanningSceneInterface()
        self.objects = []
        self.attached_objects = []
    
    def add_object(self, stl_path: str, obj_name: str, pos: NDArray, quat: NDArray):

        flag = False
        for _ in range(5):
            self._load_obj_to_moveit_scene(stl_path, pos, quat, obj_name)
            if self._check_added_to_moveit_scene(obj_name, obj_is_known=True, obj_is_attached=False):
                flag = True
                break

        if not flag:
            raise ValueError("Could not add {:s} to the moveit scene.".format(obj_name))

        self.objects.append(obj_name)

    def add_box(self, obj_name: str, pos: NDArray, quat: NDArray, size: NDArray):

        flag = False
        for _ in range(5):
            msg = rw_utils.to_stamped_pose_message(pos, quat, "base_link")
            self.scene.add_box(obj_name, msg, size)
            if self._check_added_to_moveit_scene(obj_name, obj_is_known=True, obj_is_attached=False):
                flag = True
                break

        if not flag:
            raise ValueError("Could not add {:s} to the moveit scene.".format(obj_name))

        self.objects.append(obj_name)

    def remove_object(self, obj_name: str):

        if obj_name == "" or obj_name is None:
            raise ValueError("Removing an object with an empty name would remove all objects.")
        if obj_name not in self.objects:
            raise ValueError("{:s} was not added through this interface.")

        flag = False
        for _ in range(5):
            self.scene.remove_world_object(name=obj_name)
            if self._check_added_to_moveit_scene(obj_name, obj_is_known=False, obj_is_attached=False):
                flag = True
                break
    
        if not flag:
            raise ValueError("Could not remove {:s} from moveit scene.".format(obj_name))

        self.objects.remove(obj_name)

    def attach_object(self, obj_name: str):

        flag = False
        for _ in range(5):
            self._attach_obj_to_hand(obj_name)
            # TODO: IDK
            if self._check_added_to_moveit_scene(obj_name, obj_is_known=True, obj_is_attached=True) or True:
                flag = True
                break
    
        if not flag:
            raise ValueError("Could not attach {:s} to the robot hand in moveit.".format(obj_name))
        
        self.attached_objects.append(obj_name)

    def detach_object(self, obj_name: str):

        if obj_name == "" or obj_name is None:
            raise ValueError("Detaching an object with an empty name would detach all objects.")
        if obj_name not in self.attached_objects:
            raise ValueError("{:s} was not attached through this interface.")

        flag = False
        for _ in range(5):
            self.scene.remove_attached_object(name=obj_name)
            if self._check_added_to_moveit_scene(obj_name, obj_is_known=True, obj_is_attached=False):
                flag = True
                break
    
        if not flag:
            raise ValueError("Could not detach {:s} from the robot hand in moveit.".format(obj_name))

        self.attached_objects.remove(obj_name)

    def clear(self):

        flag = False
        for _ in range(5):
            self.scene.clear()
            if self._check_clean_moveit_scene():
                flag = True
                break
    
        if not flag:
            raise ValueError("Could not clear scene.")

        self.objects = []
        self.attached_objects = []

    def _load_obj_to_moveit_scene(self, obj_path: str, pos: NDArray, quat: NDArray, obj_name: str):

        msg = rw_utils.to_stamped_pose_message(pos, quat, "base_link")
        self.scene.add_mesh(obj_name, msg, obj_path)

    def _check_added_to_moveit_scene(self, obj_name: str, timeout: int=2, obj_is_known: bool=True,
                                     obj_is_attached: bool=False) -> bool:
        """
        Set obj_is_known=False to wait for an object to be deleted.
        Set obj_is_attached=True to wait for an object to be attached to another object.
        """

        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():
            # Test if the box is in attached objects
            attached_objects = self.scene.get_attached_objects([obj_name])
            is_attached = len(attached_objects.keys()) > 0

            # Test if the box is in the scene.
            # Note that attaching the box will remove it from known_objects
            is_known = obj_name in self.scene.get_known_object_names()

            # Test if we are in the expected state
            if (obj_is_attached == is_attached) and (obj_is_known == is_known):
                return True

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        return False

    def _check_clean_moveit_scene(self, timeout: int=2) -> bool:

        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():

            if len(self.scene.get_known_object_names()) == 0:
                return True

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        return False

    def _attach_obj_to_hand(self, name: str):

        aco = AttachedCollisionObject()
        aco.object = CollisionObject()
        aco.object.id = name
        aco.link_name = "flange"
        aco.touch_links = RobotiqGripper.TOUCH_LINKS
        self.scene.attach_object(aco)
