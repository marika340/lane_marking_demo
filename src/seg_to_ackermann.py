#!/usr/bin/env python3
import os
import sys

# 1) Add the dvs_learning/models folder to Python’s import path
script_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))            # …/lane_marking_demo
models_dir  = os.path.join(project_root, 'dvs_learning', 'models')
sys.path.insert(0, models_dir)

import rospy
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge, CvBridgeError

import torch
import torch.nn.functional as F
import numpy as np

# adjust this import to wherever your model.py lives
from model import ViT  

class SegToAckermannNode:
    def __init__(self):
        rospy.init_node('seg_to_ackermann')

        # --- PARAMETERS ---
        model_path   = rospy.get_param('~model_path')            # e.g. '/home/marikan/.../model_final.pth'
        mask_topic   = rospy.get_param('~mask_topic',  'seg_mask')
        cmd_topic    = rospy.get_param('~cmd_topic',   'ackermann_cmd')
        frame_id     = rospy.get_param('~frame_id',    'base_link')

        # --- SETUP DEVICE & MODEL ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = ViT().to(self.device).eval()
        checkpoint  = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)

        # --- ROS PUB/SUB ---
        self.bridge = CvBridge()
        self.pub    = rospy.Publisher(cmd_topic, AckermannDriveStamped, queue_size=1)
        self.sub    = rospy.Subscriber(mask_topic, Image, self.mask_callback, queue_size=1)

        rospy.loginfo(f"[seg_to_ackermann] Loaded model from '{model_path}', subscribing to '{mask_topic}'")

    def mask_callback(self, img_msg: Image):
        # 1) Convert ROS Image → OpenCV grayscale
        try:
            cv_mask = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='mono8')
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")
            return

        # 2) Preprocess → Torch tensor (B=1, C=1, H, W)
        mask = cv_mask.astype(np.float32) / 255.0
        tensor_mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device)

        # 3) Forward through your ViT
        with torch.no_grad():
            pred, _ = self.model([tensor_mask])
            # pred is shape (1,2): [steering, acceleration]
            steering    = float(pred[0, 0].item())
            acceleration= float(pred[0, 1].item())

        # 4) Fill AckermannDriveStamped and publish
        cmd_msg = AckermannDriveStamped()
        cmd_msg.header.stamp    = rospy.Time.now()
        cmd_msg.header.frame_id = rospy.get_param('~frame_id', 'base_link')
        cmd_msg.drive.steering_angle   = steering
        cmd_msg.drive.acceleration     = acceleration
        # you could also set drive.speed if you predict it:
        # cmd_msg.drive.speed = some_value

        self.pub.publish(cmd_msg)

if __name__ == '__main__':
    node = SegToAckermannNode()
    rospy.spin()
