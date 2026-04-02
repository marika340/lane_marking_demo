#!/usr/bin/env python3
import rospy
import torch
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped
import os, sys
from os.path import join as opj
sys.path.append(opj(os.path.dirname(os.path.abspath(__file__)), '../dvs_learning/seg2cmd/models'))
import model as model_library

# ========== Model Wrapper ==========
class NNController:
    def __init__(self, model_path, norm_stats_path, model_type="SimpleCNN", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = getattr(model_library, model_type)(output_dim=2, return_sequence=False).to(self.device).float()
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.cmd_mean, self.cmd_std = self._load_norm_stats(norm_stats_path)
        self.bridge = CvBridge()

    def _load_norm_stats(self, path):
        cmd_mean, cmd_std = None, None
        with open(path, "r") as f:
            for line in f:
                if line.startswith("cmd_mean:"):
                    cmd_mean = np.fromstring(line.split(":")[1], sep=" ", dtype=np.float32)
                elif line.startswith("cmd_std:"):
                    cmd_std = np.fromstring(line.split(":")[1], sep=" ", dtype=np.float32)
        return cmd_mean, cmd_std

    def preprocess(self, cv_img):
        cv_img = cv2.resize(cv_img, (90, 60))   # match training resolution
        tensor = torch.from_numpy(cv_img).unsqueeze(0).unsqueeze(0).float() / 255.0
        return tensor.to(self.device)

    def predict(self, cv_img):
        tensor = self.preprocess(cv_img)
        with torch.no_grad():
            output, _ = self.model(tensor)
        steering, velocity = output[0].cpu().numpy().tolist()
        steering = steering * self.cmd_std[0] + self.cmd_mean[0]
        velocity = velocity * self.cmd_std[1] + self.cmd_mean[1]
        steering *= 5.0  # scale steering
        return steering, velocity


# ========== ROS Node ==========
class NNControllerNode:
    def __init__(self, controller):
        self.controller = controller
        self.pub = rospy.Publisher("/ackermann_drive", AckermannDriveStamped, queue_size=1)
        rospy.Subscriber("/lane_demo/front_cam/mask", Image, self.callback)

    def callback(self, msg):
        cv_img = self.controller.bridge.imgmsg_to_cv2(msg, "mono8")
        steering, velocity = self.controller.predict(cv_img)

        print(f"[ROS] Steering: {steering:.3f}, Velocity: {velocity:.3f}")
        cmd = AckermannDriveStamped()
        cmd.drive.steering_angle = steering
        cmd.drive.speed = velocity
        self.pub.publish(cmd)


# ========== Offline Test Mode ==========
def offline_test(controller, image_paths):
    for path in image_paths:
        cv_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        steering, velocity = controller.predict(cv_img)
        print(f"[OFFLINE] {path} → Steering: {steering:.3f}, Velocity: {velocity:.3f}")


def main():
    model_path = "/home/marikan/catkin_ws/src/lane_marking_demo/logs/d09_01_t22_46/nn_controller.pth"
    norm_stats_path = "/home/marikan/catkin_ws/src/lane_marking_demo/logs/d09_01_t22_46/norm_stats.txt"
    controller = NNController(model_path, norm_stats_path)

    if len(sys.argv) > 1 and sys.argv[1] == "offline":
        # Example: python nn_controller_node.py offline test1.png test2.png
        offline_test(controller, sys.argv[2:])
    else:
        rospy.init_node("nn_controller_node_debug", anonymous=False)
        NNControllerNode(controller)
        rospy.spin()


if __name__ == "__main__":
    main()
