#!/usr/bin/env python
import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient


import rospy
import time
from std_msgs.msg import Float32MultiArray

class Go2Move:
    def __init__(self):
        # g1 client
        self.sport_client = SportClient()  
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()
        time.sleep(1.0)

        # Initialize the node
        rospy.init_node('go2_move', anonymous=True)

        self.sub_humans = rospy.Subscriber("/cmd_vel", Float32MultiArray, 
                                            self.callback_cmd)

        rospy.loginfo("Go2Move initialized and waiting for messages...")

    def callback_cmd(self, msg):
        """Callback function to handle incoming Twist messages."""
        print("linear and angular velocity: ", msg.data[0], msg.data[1])
        self.sport_client.Move(msg.data[0], 0.0, msg.data[1])

    def spin(self):
        """Keep running until Ctrl+C"""
        rate = rospy.Rate(10)  # 10 Hz loop rate
        while not rospy.is_shutdown():
            rate.sleep()

    def stop(self):
        self.sport_client.StopMove()

if __name__ == "__main__":

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    ChannelFactoryInitialize(0, "eno1")
    node = Go2Move()
    try:
        node.spin()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        node.stop()
        time.sleep(2.0)

    