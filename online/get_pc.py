import rospy
from online.camera_desk import CloudProxyDesk


rospy.init_node('ur5_env')


cloud_proxy = CloudProxyDesk(desk_center=self.desk_center, z_min=self.z_min, obs_clip_offset=0, desk_offset=0.05)
