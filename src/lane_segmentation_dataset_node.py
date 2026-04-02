#!/usr/bin/env python
import rospy
import cv2
import os
from cv_bridge import CvBridge
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from datetime import datetime
from ackermann_msgs.msg import AckermannDriveStamped

class LaneSegmentationDatasetCollector:
    def __init__(self):
        rospy.init_node("lane_segmentation_dataset_node")

        # Output directory (fixed or parameterized)
        self.output_dir = os.path.expanduser("~/catkin_ws/src/lane_marking_demo/dataset")
        os.makedirs(os.path.join(self.output_dir, "events"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "masks"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "commands"), exist_ok=True)

        # CV Bridge
        self.bridge = CvBridge()
        self.latest_cmd = Twist()

        # ROS subscribers
        rospy.Subscriber("/dvs_rendering", Image, self.dvs_callback)
        rospy.Subscriber("/camera/rgb_camera/rgb/image_raw", Image, self.rgb_callback)
        rospy.Subscriber("/ackermann_drive", AckermannDriveStamped, self.ackermann_callback)

    def ackermann_callback(self, msg):
        self.latest_cmd = msg.drive

    def dvs_callback(self, msg):
        """Save the raw DVS camera image."""
        try:
            dvs_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"Error converting DVS image: {e}")
            return

        timestamp = rospy.Time.now().to_nsec()  # Unique timestamp for filenames
        event_path = os.path.join(self.output_dir, "events", f"{timestamp}.png")
        cmd_path = os.path.join(self.output_dir, "commands", f"{timestamp}.txt")

        cv2.imwrite(event_path, dvs_image)

        with open(cmd_path, "w") as f:
            # f.write(f"{self.latest_cmd.speed:.4f},{self.latest_cmd.steering_angle:.4f}\n")
            f.write(f"{self.latest_cmd}\n")

        rospy.loginfo(f"Saved DVS image and command at {timestamp}")

    def rgb_callback(self, msg):
        """Generate and save the segmentation mask from the RGB image."""
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"Error converting RGB image: {e}")
            return

        mask = self.generate_ground_truth_mask(rgb_image)

        timestamp = rospy.Time.now().to_nsec()
        mask_path = os.path.join(self.output_dir, "masks", f"{timestamp}.png")

        cv2.imwrite(mask_path, mask)

        rospy.loginfo(f"Saved segmentation mask at {timestamp}")

    def generate_ground_truth_mask(self, image):
        """Segment light gray/white lanes from the RGB image."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Laptop
        # lower_gray = np.array([0, 0, 80])
        # upper_gray = np.array([180, 50, 200])
        # mask = cv2.inRange(hsv, lower_gray, upper_gray)

        # Desktop
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

if __name__ == '__main__':
    try:
        node = LaneSegmentationDatasetCollector()
        rospy.spin()  # Keeps the node running
    except rospy.ROSInterruptException:
        pass
