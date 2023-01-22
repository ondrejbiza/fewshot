import time
import signal
import rospy
from robotiq_c_model_control.msg import CModel_robot_output as GripperCmd
from robotiq_c_model_control.msg import CModel_robot_input as GripperStat
from sensor_msgs.msg import JointState
from std_msgs.msg import Header


def main():

    signal.signal(signal.SIGINT, signal.default_int_handler)

    rospy.init_node("publish_finger_state")

    data = [0]

    def update_gripper_stat(msg):
        data[0] = msg.gPO / 255.

    gripper_sub = rospy.Subscriber("/CModelRobotInput", GripperStat, update_gripper_stat)
    joint_state_pub = rospy.Publisher('joint_states', JointState, queue_size=10)
    rate = rospy.Rate(10) # 10hz

    try:
        while True:
            hello_str = JointState()
            hello_str.header = Header()
            hello_str.header.stamp = rospy.Time.now()
            hello_str.name = ['robotiq_85_left_knuckle_joint']
            hello_str.position = [data[0]]
            hello_str.velocity = []
            hello_str.effort = []
            joint_state_pub.publish(hello_str)
            rate.sleep()
    except KeyboardInterrupt:
        gripper_sub.unregister()
        joint_state_pub.unregister()


main()
