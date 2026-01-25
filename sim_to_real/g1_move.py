#!/usr/bin/env python
import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

import rospy
from geometry_msgs.msg import Twist
import time


class TwistSubscriber:
    def __init__(self):
        # g1 client
        self.sport_client = LocoClient()  
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()
        time.sleep(1.0)

        # Initialize the node
        rospy.init_node('twist_listener', anonymous=True)

        # Create a subscriber to /cmd_vel
        self.sub = rospy.Subscriber('/cmd_vel', Twist, self.callback)

        # Store the latest twist message
        self.latest_twist = Twist()

        rospy.loginfo("TwistSubscriber initialized and waiting for messages...")

    def callback(self, msg):
        """Callback function to handle incoming Twist messages."""
        print("linear and angular velocity: ", msg.linear.x, msg.angular.z)
        self.sport_client.Move(msg.linear.x, 0.0, msg.angular.z)

    def spin(self):
        """Keep running until Ctrl+C"""
        rate = rospy.Rate(10)  # 10 Hz loop rate
        while not rospy.is_shutdown():
            # You could process or print the latest_twist here if needed
            rate.sleep()

    def stop(self):
        self.sport_client.StopMove()

if __name__ == "__main__":

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    ChannelFactoryInitialize(0, "enx9c69d342225b")
    node = TwistSubscriber()
    try:
        node.spin()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        node.stop()
        time.sleep(2.0)

    