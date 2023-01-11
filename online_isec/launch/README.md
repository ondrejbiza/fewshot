Start:

roslaunch ur_robot_driver ur5_bringup.launch robot_ip:=10.75.15.199 limited:=true headless_mode:=true

sudo chmod 777 /dev/ttyUSB0
rosrun robotiq_c_model_control CModelRtuNode.py /dev/ttyUSB0

rostopic hz /camera/depth/points
rostopic hz /camera/ir/image
rostopic hz /cam1/depth/color/points
rostopic hz /k4a/depth_registered/points

cd ~/catkin_ws/src/fewshot/online_isec/launch
roslaunch openni2_launch openni2.launch
roslaunch realsense.launch
roslaunch azure.launch
python add_sensor_frame.py 

Calibrate:

Stop python add_sensor_frame.py.
Optionally set <arg name="ordered_pc" value="false"/> in realsense.launch to speed up rviz.

tab1:
roslaunch ur5_moveit_config ur5_moveit_planning_execution.launch

tab2:
roslaunch ur5_moveit_config moveit_rviz.launch config:=true

tab3:
sudo su
source /home/ur5/.bashrc
python keyboard_py.py base_link cam1_color_optical_frame 0.513 0.551 0.767 quaternion -0.2546 0.6516 -0.2704 0.6615 50

# stack

Left: structure sensor
Middle: realsense 455
Right: Azure
TODO: recalibrate realsense -- previously calibrated in the frame of depth, now in the frame of RGB (90% sure)
https://github.com/IntelRealSense/realsense-ros/tree/ros1-legacy

## Requirements
* fmauch_universal_robot
* robotiq
* Universal_Robots_ROS_Driver
* Catkin make with python 3.7
   ```   
   catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3.7
   ```
## Dependencies


### Instructions
1. Start UR driver.
    ```
    roslaunch ur_robot_driver ur5_bringup.launch robot_ip:=10.75.15.199 limited:=true headless_mode:=true
    ```
2. Open subscriber
    ```
   rostopic hz /k4a/depth_registered/points
   rostopic hz /cam1/depth/color/points
   rostopic hz /camera/
   ```
3. Start gripper driver.
    ```
   sudo chmod 777 /dev/ttyUSB0
   rosrun robotiq_c_model_control CModelRtuNode.py /dev/ttyUSB0
    ```

4. Start depth sensor driver.
    ```
   cd ~/catkin_ws/src/fewshot/online_isec/launch
   roslaunch openni2_launch openni2.launch
   roslaunch azure.launch
   roslaunch realsense.launch
   python add_sensor_frame.py 
    ```
5. Example for openni2 launch file
   ```
   <launch>
  <!-- launch up sensor-->
  <include file="$(find openni2_launch)/launch/openni2.launch">
    <arg name="camera" value="camera_up" />
    <arg name="device_id" value="1d27/0600@11"/>
  </include>
   </launch>
   ```
## Troubleshooting
1. After rebooting, gain the access for the USB port for the gripper:
   ```
   sudo chmod 777 /dev/ttyUSB0
   ```
1. Install python3 catkin packages:
   ```
   sudo apt-get install python-catkin-pkg
   sudo apt-get install python3-catkin-pkg-modules
   sudo apt-get install python3-rospkg-modules
   ```
   https://answers.ros.org/question/245967/importerror-no-module-named-rospkg-python3-solved/?answer=298221#post-id-298221
1. Opening openni2 driver: ResourceNotFound: rgbd_launch:
   ```
   sudo apt install ros-kinetic-rgbd-launch
   ```
1. Could not find "controller interface"
   ```
   sudo apt-get install ros-kinetic-ros-control ros-kinetic-ros-controllers
   ```