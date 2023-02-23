import argparse
import os
import subprocess
import signal
import sys
import time

ROBOT_LOG_PATH = "logs/robot.txt"

REALSENSE_LEFT_LISTEN_LOG_PATH = "logs/realsense_left_listen.txt"
REALSENSE_RIGHT_LISTEN_LOG_PATH = "logs/realsense_right_listen.txt"
REALSENSE_FORWARD_LISTEN_LOG_PATH = "logs/realsense_forward_listen.txt"

REALSENSE_LEFT_DRIVER_LOG_PATH = "logs/realsense_left_driver.txt"
REALSENSE_RIGHT_DRIVER_LOG_PATH = "logs/realsense_right_driver.txt"
REALSENSE_FORWARD_DRIVER_LOG_PATH = "logs/realsense_forward_driver.txt"

GRIPPER_LOG_PATH = "logs/gripper.txt"
GRIPPER_PUB_LOG_PATH = "logs/gripper_pub.txt"
ADD_SENSOR_FRAME_LOG_PATH = "logs/add_sensor_frame.txt"
PLANNING_LOG_PATH = "logs/planning.txt"
RVIZ_LOG_PATH = "logs/rviz.txt"

# TODO: Not sure why I don't get other outputs.
ROBOT_SUCCESS_PHRASE = "No realtime capabilities found. Consider using a realtime system for better performance"
ROBOT_CONNECTION_FAILURE_PHRASE = "Connection setup failed"

class FailureException(Exception):
    pass


def check_robot_started():

    ok = False
    connection_failure = False

    with open(ROBOT_LOG_PATH, "r") as f:
        for line in f.readlines():
            if ROBOT_SUCCESS_PHRASE in line:
                ok = True
            if ROBOT_CONNECTION_FAILURE_PHRASE in line:
                connection_failure = True

    return ok, connection_failure


def close_log(log):
    log.close()


def close_process(p):
    p.send_signal(signal.SIGINT)
    p.wait()


def kill_process(p):
    p.send_signal(signal.SIGKILL)
    p.wait()


def main(args):

    # Every process and log we have to open and close.
    robot_log = None
    robot_p = None

    gripper_log = None
    gripper_p = None

    gripper_pub_log = None
    gripper_pub_p = None

    realsense_listen_log_list = []
    realsense_listen_p_list = []
    realsense_driver_log_list = []
    realsense_driver_p_list = []

    add_sensor_frame_log = None
    add_sensor_frame_p = None

    planning_log = None
    planning_p = None

    rviz_log = None
    rviz_p = None

    # Proper ctrl+C handling.
    signal.signal(signal.SIGINT, signal.default_int_handler)

    if not os.path.isdir("logs"):
        os.makedirs("logs")

    try:
        # Connect robot.
        print("Connecting to UR5 at {:s}.".format(args.ip))
        robot_log = open(ROBOT_LOG_PATH, "w")
        robot_p = subprocess.Popen(
            " ".join(["roslaunch", "ur_robot_driver", "ur5_bringup.launch",
            "robot_ip:={:s}".format(args.ip), "headless_mode:=true",
            "kinematics_config:=$(rospack find ur_calibration)/etc/ur5_isec_calibration.yaml"]),  # limited:=true if motion planning doesn't work
            stdout=robot_log, stderr=subprocess.STDOUT, shell=True
        )

        i = 0
        while True:
            time.sleep(1)
            ok, connection_failure = check_robot_started()
            if connection_failure:
                print("Wrong UR5 IP address.")
                print("Go to the 'About' page on the UR5 tablet to find the right one.")
                raise FailureException()
            if ok:
                break
            if i > 30:
                print("Robot is taking too long to start. ")
            i += 1
        print("UR5 connected.")
        time.sleep(1)

        # Connect gripper.
        print("Connecting gripper.")
        code = subprocess.call(["sudo", "chmod", "777", "/dev/ttyUSB0"])
        if code != 0:
            print("Expecting gripper at /dev/ttyUSB0.")
            raise FailureException()
        gripper_log = open(GRIPPER_LOG_PATH, "w")
        gripper_p = subprocess.Popen(
            ["rosrun", "robotiq_c_model_control", "CModelRtuNode.py", "/dev/ttyUSB0"],
            stdout=gripper_log, stderr=subprocess.STDOUT
        )

        gripper_pub_log = open(GRIPPER_PUB_LOG_PATH, "w")
        gripper_pub_p = subprocess.Popen(
            ["python", "publish_finger_state.py"],
            stdout=gripper_pub_log, stderr=subprocess.STDOUT
        )
        print("Gripper connected.")

        print("Connecting cameras.")
        for log_path, list_name in zip(
            [REALSENSE_LEFT_LISTEN_LOG_PATH, REALSENSE_RIGHT_LISTEN_LOG_PATH, REALSENSE_FORWARD_LISTEN_LOG_PATH],
            ["/realsense_left/depth/color/points", "/realsense_right/depth/color/points", "/realsense_forward/depth/color/points"]
        ):
            realsense_listen_log_list.append(open(log_path, "w"))
            realsense_listen_p_list.append(subprocess.Popen(
                ["rostopic", "hz", list_name],
                stdout=realsense_listen_log_list[-1], stderr=subprocess.STDOUT
            ))

        for log_path, launch_name in zip(
                [REALSENSE_LEFT_DRIVER_LOG_PATH, REALSENSE_RIGHT_DRIVER_LOG_PATH, REALSENSE_FORWARD_DRIVER_LOG_PATH],
                ["realsense_left.launch", "realsense_right.launch", "realsense_forward.launch"]
            ):
            realsense_driver_log_list.append(open(log_path, "w"))
            realsense_driver_p_list.append(subprocess.Popen(
                ["roslaunch", launch_name],
                stdout=realsense_driver_log_list[-1], stderr=subprocess.STDOUT
            ))
            time.sleep(5)

        add_sensor_frame_log = open(ADD_SENSOR_FRAME_LOG_PATH, "w")
        add_sensor_frame_p = subprocess.Popen(
            ["python", "add_sensor_frame.py"],
            stdout=add_sensor_frame_log, stderr=subprocess.STDOUT
        )
        print("Cameras connected.")

        print("Connecting planning.")
        planning_log = open(PLANNING_LOG_PATH, "w")
        planning_p = subprocess.Popen(
            ["roslaunch", "ur5_robotiq_moveit_config", "move_group.launch"],
            stdout=planning_log, stderr=subprocess.STDOUT
        )
        print("Planning connected.")

        print("Starting rviz.")
        dir_path = os.path.dirname(os.path.realpath(__file__))
        rviz_log = open(RVIZ_LOG_PATH, "w")
        rviz_p = subprocess.Popen(
            ["roslaunch", "ur5_moveit_config", "moveit_rviz.launch", "rviz_config:={:s}".format(os.path.join(dir_path, "rviz_config.rviz"))],
            stdout=rviz_log, stderr=subprocess.STDOUT
        )
        print("rviz started.")

        while True:
            time.sleep(1)

    except (KeyboardInterrupt, FailureException):
        print("Exitting.")

        print("Closing UR5 connection.")
        if robot_p is not None:
            close_process(robot_p)
        if robot_log is not None:
            close_log(robot_log)
        print("UR5 connection closed.")

        print("Closing gripper connection.")
        if gripper_p is not None:
            close_process(gripper_p)
        if gripper_log is not None:
            close_log(gripper_log)

        if gripper_pub_p is not None:
            close_process(gripper_pub_p)
        if gripper_pub_log is not None:
            close_log(gripper_pub_log)
        print("Gripper connection closed.")

        print("Closing camera connection.")
        for x in realsense_listen_p_list:
            close_process(x)
        for x in realsense_listen_log_list:
            close_log(x)

        for x in realsense_driver_p_list:
            close_process(x)
        for x in realsense_driver_log_list:
            close_log(x)

        if add_sensor_frame_p is not None:
            close_process(add_sensor_frame_p)
        if add_sensor_frame_log is not None:
            close_log(add_sensor_frame_log)
        print("Camera connection closed.")

        print("Closing planning.")
        if planning_p is not None:
            close_process(planning_p)
        if planning_log is not None:
            close_log(planning_log)
        print("Planning closed.")

        print("Closing rviz.")
        if rviz_p is not None:
            kill_process(rviz_p)
        if rviz_log is not None:
            close_log(rviz_log)
        print("rviz closed.")

        sys.exit(0)


parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="10.75.15.199")
main(parser.parse_args())
