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
    # Hand calibration. Use this if align_depth is false in the realsense.launch config.
    # br.sendTransform((0.778, -0.094, 0.431), (0.4147, -0.0251, -0.9096, -0.0023), rospy.Time.now(), "cam1_depth_frame", "base_link")
    # Hand calibration.
    # br.sendTransform((0.779, -0.022, 0.43), (-0.6558, -0.6589, 0.2725, 0.248), rospy.Time.now(), "cam1_color_optical_frame", "base_link")  # TODO: calibrate
    # Automatic calibration.
    br.sendTransform((-0.78480323, 0.04703094, 0.44310847), (-0.65515551, 0.66065954, -0.24920589, 0.26869435), rospy.Time.now(), "cam1_color_optical_frame", "base")
  
    # azure
    # Hand calibration.
    br.sendTransform((0.503, 0.547, 0.762), (-0.2611, 0.649, -0.2638, 0.6641), rospy.Time.now(), "camera_depth_frame", "base_link")
    # Automatic calibration. Doesn't work.
    # br.sendTransform((-0.54775263, -0.04869692, 0.92453691), (0.93359084, -0.00268021, 0.35799047, 0.0156135), rospy.Time.now(), "camera_depth_frame", "base")

    # structure
    br.sendTransform((0.691, -0.833, 0.018), (0.035, 0.664, 0.746, 0.035), rospy.Time.now(), "depth_camera_link", "camera_depth_frame")

    rate.sleep()
