#!/usr/bin/env python
'''Broadcasts the transform for the sensor.

Credits:
- http://wiki.ros.org/tf/Tutorials/Adding%20a%20frame%20%28Python%29

Assembled: Northeastern University, 2015
'''

import rospy
import tf

if __name__ == '__main__':
  
  rospy.init_node('add_sensor_frame')
  br = tf.TransformBroadcaster()
  rate = rospy.Rate(50.0)
  
  while not rospy.is_shutdown():
    # measured
    # br.sendTransform((0.092, 0.062, 0.044), (0, 0, 0, 1), rospy.Time.now(), "camera_link", "ee_link")
    # block calibrated
    # br.sendTransform((0.094, 0.061, 0.050), (0, 0, 0, 1), rospy.Time.now(), "camera_link", "ee_link")
    # br.sendTransform((0.096, 0.061, 0.046), (0, 0, 0, 1), rospy.Time.now(), "camera_link", "ee_link")
    # br.sendTran sform((0.08, 0.061, 0.046), (0, 0, 0, 1), rospy.Time.now(), "camera_link", "ee_link")

    # br.sendTransform((0.52, 0.546, 0.771), (-0.2579, 0.6608, -0.2671, 0.6523), rospy.Time.now(), "camera_depth_frame", "base_link")
    # br.sendTransform((0.479, -0.607, 0.778), (-0.6681, -0.2564, 0.6307, 0.3001), rospy.Time.now(), "cam1_depth_frame", "base_link")

    # br.sendTransform((0.513, 0.551, 0.767), (-0.2546, 0.6516, -0.2704, 0.6615), rospy.Time.now(), "camera_depth_frame", "base_link")
    # br.sendTransform((0.493, -0.604, 0.776), (-0.6642, -0.2664, 0.6351, 0.2906), rospy.Time.now(), "cam1_depth_frame", "base_link")
    # br.sendTransform((0.513, -0.529, 0.841), (0.6798, 0.6427, 0.2918, -0.199), rospy.Time.now(), "depth_camera_link", "base_link")

    # realsense
    # Automatic calibration with images.
    # br.sendTransform((-0.779918, 0.043688, 0.449073), (-0.656797, 0.662001, -0.243188, 0.266894), rospy.Time.now(), "cam1_color_optical_frame", "base")
    # Automatic calibration with depths.
    # br.sendTransform((-0.78302526, 0.0488187, 0.44769223), (-0.6538833, 0.66373159, -0.24695839, 0.26628662), rospy.Time.now(), "cam1_color_optical_frame", "base")
    # Automatic calibration with depths. Fine-tuned by hand.
    br.sendTransform((-0.774, 0.0378, 0.4377), (-0.6538833, 0.66373159, -0.24695839, 0.26628662), rospy.Time.now(), "cam1_color_optical_frame", "base")

    # azure
    # Automatic calibration with images.
    # TODO: only works for /k4a/depth/points, not /k4a/depth_registered/points
    br.sendTransform((-0.499851, 0.545556, 0.854526), (0.658480, -0.656506, 0.242301, 0.276939), rospy.Time.now(), "rgb_camera_link", "base")
    # TODO: somehow I need to calibrate this
    br.sendTransform((0., 0., 0.), (1., 0., 0., 0.), rospy.Time.now(), "depth_camera_link", "rgb_camera_link")

    # structure
    # Automatic calibration with images.
    # br.sendTransform((-0.479379, -0.564626, 0.776313), (0.656418, 0.650506, 0.254785, -0.284680), rospy.Time.now(), "camera_depth_optical_frame", "base")
    # Automatic calibration with depths.
    # br.sendTransform((-0.50859704, -0.55556215, 0.77484579), (0.65841246, 0.65057976, 0.262971, -0.27218609), rospy.Time.now(), "camera_depth_optical_frame", "base")
    # Automatic calibration with depths. Fine-tuned by hand.
    br.sendTransform((-0.5056, -0.5606, 0.7568), (0.65841246, 0.65057976, 0.262971, -0.27218609), rospy.Time.now(), "camera_depth_optical_frame", "base")

    rate.sleep()
