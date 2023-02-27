from geometry_msgs.msg import PoseStamped
from numpy.typing import NDArray
from moveit_msgs.msg import DisplayTrajectory
import rospy
from visualization_msgs.msg import Marker

import src.real_world.utils as rw_utils


class RVizPub:
    """Send visualizations to rviz as ROS messages."""

    def __init__(self):
        self.marker_pub = rospy.Publisher("visualization_marker", Marker)
        self.trajectory_pub = rospy.Publisher("/move_group/display_planned_path", DisplayTrajectory)
        self.pose_pub = rospy.Publisher("pose_publisher", PoseStamped)

    def send_stl_message(self, stl_path: str, pos: NDArray, quat: NDArray):
        """Visualize an STL object."""
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.MESH_RESOURCE
        marker.mesh_resource = stl_path
        marker.action = Marker.ADD
        marker.pose = rw_utils.to_pose_message(pos, quat)
        marker.scale.x = 1.
        marker.scale.y = 1.
        marker.scale.z = 1.
        marker.color.r = 0.
        marker.color.g = 0.
        marker.color.b = 1.
        marker.color.a = 1.

        self.marker_pub.publish(marker)

    def send_trajectory_message(self, ds: DisplayTrajectory):
        """Visualize trajectory."""
        self.trajectory_pub.publish(ds)

    def send_pose(self, pos: NDArray, quat: NDArray, frame_id: str):
        """Visualize pose."""
        pose = rw_utils.to_stamped_pose_message(pos, quat, frame_id)
        self.pose_pub.publish(pose)
