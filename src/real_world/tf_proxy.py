from numpy.typing import NDArray
import rospy
import tf2_ros

from src import utils


class TFProxy:

    def __init__(self):

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def lookup_transform(self, source_frame: str, target_frame: str, lookupTime: rospy.Time=rospy.Time(0)) -> NDArray:
        """
        Lookup a transform in the TF tree.
        :param source_frame: the frame from which the transform is calculated
        :param target_frame: the frame to which the transform is calculated
        :return: transformation matrix from fromFrame to toFrame
        :rtype: 4x4 np.array
        """
        transformMsg = self.tf_buffer.lookup_transform(target_frame, source_frame, lookupTime, rospy.Duration(1.0))
        translation = transformMsg.transform.translation

        pos = (translation.x, translation.y, translation.z)
        rotation = transformMsg.transform.rotation
        quat = (rotation.x, rotation.y, rotation.z, rotation.w)

        return utils.pos_quat_to_transform(pos, quat)
