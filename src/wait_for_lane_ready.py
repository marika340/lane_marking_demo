#!/usr/bin/env python
import rospy
from std_msgs.msg import Bool

def callback(msg):
    if msg.data:
        rospy.signal_shutdown("Lane ready")

rospy.init_node("wait_for_lane_ready")
rospy.Subscriber("/lane_demo/ready", Bool, callback)
rospy.spin()