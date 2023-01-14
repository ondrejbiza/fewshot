import rospy
import tf2_ros
from numpy.typing import NDArray
import online_isec.transformation as transformation


class TFProxy:

    def __init__(self):

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def lookup_transform(self, fromFrame: str, toFrame: str, lookupTime: rospy.Time=rospy.Time(0)) -> NDArray:
        """
        Lookup a transform in the TF tree.
        :param fromFrame: the frame from which the transform is calculated
        :type fromFrame: string
        :param toFrame: the frame to which the transform is calculated
        :type toFrame: string
        :return: transformation matrix from fromFrame to toFrame
        :rtype: 4x4 np.array
        """
        # TODO: I might need to swap toFrame and fromFrame for the robot.
        transformMsg = self.tf_buffer.lookup_transform(toFrame, fromFrame, lookupTime, rospy.Duration(1.0))
        translation = transformMsg.transform.translation
        pos = [translation.x, translation.y, translation.z]
        rotation = transformMsg.transform.rotation
        quat = [rotation.x, rotation.y, rotation.z, rotation.w]
        T = transformation.quaternion_matrix(quat)
        T[0:3, 3] = pos
        return T
