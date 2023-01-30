from numpy.typing import NDArray
import moveit_commander

import online_isec.utils as isec_utils


class MoveItScene:

    def __init__(self):

        self.scene = moveit_commander.PlanningSceneInterface()
        self.objects = []
        self.attached_objects = []
    
    def add_object(self, stl_path: str, obj_name: str, pos: NDArray, quat: NDArray):

        flag = False
        for _ in range(5):
            isec_utils.load_obj_to_moveit_scene_2(stl_path, pos, quat, obj_name, self.scene)
            if isec_utils.check_added_to_moveit_scene(obj_name, self.scene, obj_is_known=True, obj_is_attached=False):
                flag = True
                break

        if not flag:
            raise ValueError("Could not add {:s} to the moveit scene.".format(obj_name))

        self.objects.append(obj_name)

    def add_box(self, obj_name: str, pos: NDArray, quat: NDArray, size: NDArray):

        flag = False
        for _ in range(5):
            msg = isec_utils.to_stamped_pose_message(pos, quat, "base_link")
            self.scene.add_box(obj_name, msg, size)
            if isec_utils.check_added_to_moveit_scene(obj_name, self.scene, obj_is_known=True, obj_is_attached=False):
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
            if isec_utils.check_added_to_moveit_scene(obj_name, self.scene, obj_is_known=False, obj_is_attached=False):
                flag = True
                break
    
        if not flag:
            raise ValueError("Could not remove {:s} from moveit scene.".format(obj_name))

        self.objects.remove(obj_name)

    def attach_object(self, obj_name: str):

        flag = False
        for _ in range(5):
            isec_utils.attach_obj_to_hand(obj_name, self.scene)
            # TODO: IDK
            if isec_utils.check_added_to_moveit_scene(obj_name, self.scene, obj_is_known=True, obj_is_attached=True) or True:
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
            if isec_utils.check_added_to_moveit_scene(obj_name, self.scene, obj_is_known=True, obj_is_attached=False):
                flag = True
                break
    
        if not flag:
            raise ValueError("Could not detach {:s} from the robot hand in moveit.".format(obj_name))

        self.attached_objects.remove(obj_name)

    def clear(self):

        flag = False
        for _ in range(5):
            self.scene.clear()
            if isec_utils.check_clean_moveit_scene(self.scene):
                flag = True
                break
    
        if not flag:
            raise ValueError("Could not clear scene.")

        self.objects = []
        self.attached_objects = []
