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
  
    # br.sendTransform(
    #   (-0.5067, -0.6141, 0.1185),
    #   (-0.7919011836365644, 0.0005438989185203736, -0.014687531206070251, 0.6104723547844289),
    #   rospy.Time.now(),
    #   "realsense_left_color_optical_frame",
    #   "base"
    # )

    # br.sendTransform(
    #   (0.5691, -0.5734, 0.1142),
    #   (-0.74048394,  0.02918226, -0.0176743 ,  0.67120753),
    #   rospy.Time.now(),
    #   "realsense_right_color_optical_frame",
    #   "base"
    # )

    br.sendTransform((-0.536652, 0.023152, 0.194550), (0.563893, -0.556389, 0.438239, -0.424738), rospy.Time.now(), "realsense_left_color_optical_frame", "checkerboard")
    br.sendTransform((0.648942, 0.121316, 0.181432), (0.534039, 0.511545, -0.472061, -0.479877), rospy.Time.now(), "realsense_right_color_optical_frame", "checkerboard")
    br.sendTransform((0.0987, 0.6369, 0.2249), (-0.00582917, 0.73763816, -0.67516426, 0.00303082), rospy.Time.now(), "realsense_forward_color_optical_frame", "checkerboard")

    br.sendTransform((0.475, 0.078, -0.078), (0., 0., -0.70611744, 0.70809474), rospy.Time.now(), "checkerboard", "base_link")

    rate.sleep()
