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

# Load your trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = torch.load("/home/marikan/catkin_ws/src/lane_marking_demo/logs/d08_22_t18_31/nn_controller.pth", map_location=device)
model_type = "ViT"
model = getattr(model_library, model_type)(output_dim=2, return_sequence=True).to(device).float()
# state_dict = torch.load("/home/marikan/catkin_ws/src/lane_marking_demo/logs/d08_22_t18_31/nn_controller.pth", map_location=device)
# state_dict = torch.load("/home/marikan/catkin_ws/src/lane_marking_demo/logs/d08_25_t10_57/nn_controller_2.pth", map_location=device)
# state_dict = torch.load("/home/marikan/catkin_ws/src/lane_marking_demo/logs/d08_27_t15_08/nn_controller.pth", map_location=device)
state_dict = torch.load("/home/marikan/catkin_ws/src/lane_marking_demo/logs/d09_01_t22_46/nn_controller.pth", map_location=device)
# state_dict = torch.load("/home/marikan/catkin_ws/src/lane_marking_demo/logs/d10_07_t09_33/best_model_050.pth", map_location=device)
# state_dict = torch.load("/home/marikan/catkin_ws/src/lane_marking_demo/logs/d10_15_t14_36/best_model_054.pth", map_location=device)
# state_dict = torch.load("/home/marikan/catkin_ws/src/lane_marking_demo/logs/d10_20_t00_35/best_model_061.pth", map_location=device) # ViT
# state_dict = torch.load("/home/marikan/catkin_ws/src/lane_marking_demo/dvs_learning/event2cmd/training/log/joint_d11_05_t15_05/best_joint_326.pth", map_location=device) # ViT, event2cmd
model.load_state_dict(state_dict)
norm_stats_path = "/home/marikan/catkin_ws/src/lane_marking_demo/logs/d09_01_t22_46/norm_stats.txt"
# norm_stats_path = "/home/marikan/catkin_ws/src/lane_marking_demo/logs/d10_15_t14_36/norm_stats.txt"
# norm_stats_path = "/home/marikan/catkin_ws/src/lane_marking_demo/dvs_learning/event2cmd/training/log/joint_d11_05_t15_05/norm_stats.txt"
model.eval()

bridge = CvBridge()
pub = None

hidden_state = None

def load_norm_stats(path):
    cmd_mean, cmd_std = None, None
    with open(path, "r") as f:
        for line in f:
            if line.startswith("cmd_mean:"):
                parts = line.strip().split(":")[1].split()
                cmd_mean = np.array([float(x) for x in parts], dtype=np.float32)
            elif line.startswith("cmd_std:"):
                parts = line.strip().split(":")[1].split()
                cmd_std = np.array([float(x) for x in parts], dtype=np.float32)
    return cmd_mean, cmd_std

cmd_mean, cmd_std = load_norm_stats(norm_stats_path)

def preprocess_mask(mask_msg):
    cv_img = bridge.imgmsg_to_cv2(mask_msg, "mono8")  # segmentation mask
    cv_img = cv2.resize(cv_img, (90, 60))             # match training resolution
    # make all values either 0 or 255
    cv_img = np.where(cv_img > 20, 255, 0).astype(np.uint8)
    print(f'[process_mask] mask shape: {cv_img.shape}, unique values: {np.unique(cv_img)}')
    tensor = torch.from_numpy(cv_img).unsqueeze(0).unsqueeze(0).float() / 255.0
    print(f'[process_mask] tensor shape: {tensor.shape}, min: {tensor.min()}, max: {tensor.max()}')
    return tensor.to(device)

def mask_callback(msg):
    print('[mask_callback] Received mask message')
    global hidden_state
    mask_tensor = preprocess_mask(msg)
    with torch.no_grad():
        if model_type in ["LSTM", "LSTMViT"]:
            output, hidden_state = model([mask_tensor], hidden_state)
        else:
            output, _ = model([mask_tensor])
    print(f'[mask_callback] output shape: {output.shape}')
    steering, velocity = output[0].cpu().numpy().tolist()  # assuming 2 outputs

    print(f'Raw output - Steering: {steering}, Velocity: {velocity}')
    print(f'Cmd mean: {cmd_mean}, Cmd std: {cmd_std}')

    steering = steering * cmd_std[0] + cmd_mean[0]
    velocity = velocity * cmd_std[1] + cmd_mean[1]

    steering *= 1.0 # scale steering # 5.0 for CNN, 3.0 for LSTMViT
    
    print(f"Steering: {steering:.3f}, Velocity: {velocity:.3f}")
    

    cmd = AckermannDriveStamped()
    cmd.drive.steering_angle = steering
    cmd.drive.speed = velocity
    pub.publish(cmd)

def main():
    global pub
    rospy.init_node("nn_controller", anonymous=False)
    rospy.Subscriber("/lane_demo/front_cam/mask", Image, mask_callback)
    pub = rospy.Publisher("/ackermann_drive", AckermannDriveStamped, queue_size=1)
    rospy.spin()

if __name__ == "__main__":
    main()
