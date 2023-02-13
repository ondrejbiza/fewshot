import argparse
import os
import subprocess
import signal
import sys
import time

ROBOT_LOG_PATH = "logs/robot.txt"
AZURE_LISTEN_LOG_PATH = "logs/azure_listen.txt"
REALSENSE_LISTEN_LOG_PATH = "logs/realsense_listen.txt"
STRUCTURE_LISTEN_LOG_PATH = "logs/structure_listen.txt"
GRIPPER_LOG_PATH = "logs/gripper.txt"
GRIPPER_PUB_LOG_PATH = "logs/gripper_pub.txt"
STRUCTURE_DRIVER_LOG_PATH = "logs/structure_driver.txt"
AZURE_DRIVER_LOG_PATH = "logs/azure_driver.txt"
REALSENSE_DRIVER_LOG_PATH = "logs/realsense_driver.txt"
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

    structure_listen_log = None
    structure_listen_p = None

    realsense_listen_log = None
    realsense_listen_p = None

    # azure_listen_log = None
    # azure_listen_p = None

    structure_driver_log = None
    structure_driver_p = None

    realsense_driver_log = None
    realsense_driver_p = None

    # azure_driver_log = None
    # azure_driver_p = None

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
            ["roslaunch", "ur_robot_driver", "ur5_bringup.launch",
            "robot_ip:={:s}".format(args.ip), "headless_mode:=true"],  # limited:=true if motion planning doesn't work
            stdout=robot_log, stderr=subprocess.STDOUT
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
        time.sleep(1)

        print("Connecting cameras.")

        structure_listen_log = open(STRUCTURE_LISTEN_LOG_PATH, "w")
        structure_listen_p = subprocess.Popen(
            ["rostopic", "hz", "/camera/depth/points"],
            stdout=structure_listen_log, stderr=subprocess.STDOUT
        )

        realsense_listen_log = open(REALSENSE_LISTEN_LOG_PATH, "w")
        realsense_listen_p = subprocess.Popen(
            ["rostopic", "hz", "/cam1/depth/color/points"],
            stdout=realsense_listen_log, stderr=subprocess.STDOUT
        )

        # azure_listen_log = open(AZURE_LISTEN_LOG_PATH, "w")
        # azure_listen_p = subprocess.Popen(
        #     ["rostopic", "hz", "/k4a/points2"],
        #     stdout=azure_listen_log, stderr=azure_listen_log
        # )
        # time.sleep(1)

        structure_driver_log = open(STRUCTURE_DRIVER_LOG_PATH, "w")
        structure_driver_p = subprocess.Popen(
            ["roslaunch", "openni2_launch", "openni2.launch"],
            stdout=structure_driver_log, stderr=subprocess.STDOUT
        )
        time.sleep(5)

        realsense_driver_log = open(REALSENSE_DRIVER_LOG_PATH, "w")
        realsense_driver_p = subprocess.Popen(
            ["roslaunch", "realsense.launch"],
            stdout=realsense_driver_log, stderr=subprocess.STDOUT
        )
        time.sleep(5)

        # azure_driver_log = open(AZURE_DRIVER_LOG_PATH, "w")
        # azure_driver_p = subprocess.Popen(
        #     ["roslaunch", "azure.launch"],
        #     stdout=azure_driver_log, stderr=azure_driver_log
        # )
        # time.sleep(1)

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

        if structure_listen_p is not None:
            close_process(structure_listen_p)
        if structure_listen_log is not None:
            close_log(structure_listen_log)

        if realsense_listen_p is not None:
            close_process(realsense_listen_p)
        if realsense_listen_log is not None:
            close_log(realsense_listen_log)

        # if azure_listen_p is not None:
        #     close_process(azure_listen_p)
        # if azure_listen_log is not None:
        #     close_log(azure_listen_log)

        if structure_driver_p is not None:
            close_process(structure_driver_p)
        if structure_driver_log is not None:
            close_log(structure_driver_log)

        if realsense_driver_p is not None:
            close_process(realsense_driver_p)
        if realsense_driver_log is not None:
            close_log(realsense_driver_log)

        # if azure_driver_p is not None:
        #     close_process(azure_driver_p)
        # if azure_driver_log is not None:
        #     close_log(azure_driver_log)

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
