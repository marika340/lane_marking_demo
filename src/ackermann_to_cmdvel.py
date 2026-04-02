#!/usr/bin/env python3
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Twist
import math

WHEELBASE = 0.4

def callback(msg):
    twist = Twist()
    v = msg.drive.speed
    delta = msg.drive.steering_angle

    twist.linear.x = v
    # twist.angular.z = msg.drive.steering_angle
    # kinematic bicycle model
    twist.angular.z = v / WHEELBASE * math.tan(delta)

    pub.publish(twist)

if __name__ == "__main__":
    rospy.init_node("ackermann_to_cmdvel")
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    rospy.Subscriber("/ackermann_drive", AckermannDriveStamped, callback)
    rospy.loginfo("Ackermann → cmd_vel bridge started")
    rospy.spin()
