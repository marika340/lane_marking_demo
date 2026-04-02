#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class MaskPublisher:
    def __init__(self):
        rospy.init_node("mask_publisher", anonymous=False)
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber("/camera/rgb_camera/rgb/image_raw", Image, self.rgb_callback)
        self.pub = rospy.Publisher("/lane_demo/front_cam/mask", Image, queue_size=1)
        rospy.loginfo("MaskPublisher node started, subscribing to /camera/rgb_camera/rgb/image_raw")

    def generate_mask(self, image):
        # Same method as in your dataset collector
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        kernel = np.ones((1, 1), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # print(f'[generate_mask] mask shape: {mask.shape}, unique values: {np.unique(mask)}')

        return mask

    def rgb_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"cv_bridge error: {e}")
            return

        mask = self.generate_mask(cv_img)

        mask_msg = self.bridge.cv2_to_imgmsg(mask, "mono8")
        mask_msg.header = msg.header  # keep timestamp
        self.pub.publish(mask_msg)

if __name__ == "__main__":
    node = MaskPublisher()
    rospy.spin()
