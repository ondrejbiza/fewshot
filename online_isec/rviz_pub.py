import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import DisplayTrajectory

import online_isec.utils as isec_utils


class RVizPub:

    def __init__(self):
        self.marker_pub = rospy.Publisher("visualization_marker", Marker)
        self.pose_pub = rospy.Publisher  # TODO
        self.trajectory_pub = rospy.Publisher("/move_group/display_planned_path", DisplayTrajectory, queue_size=20)

    def send_stl_message(self, stl_path, pos, quat):

        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.MESH_RESOURCE
        marker.mesh_resource = stl_path
        marker.action = Marker.ADD
        marker.pose = isec_utils.to_pose_message(pos, quat)
        marker.scale.x = 1.
        marker.scale.y = 1.
        marker.scale.z = 1.
        marker.color.r = 0.
        marker.color.g = 0.
        marker.color.b = 1.
        marker.color.a = 1.

        self.marker_pub.publish(marker)

    def send_trajectory_message(self, ds: DisplayTrajectory):
        self.trajectory_pub.publish(ds)
