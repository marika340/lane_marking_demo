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
# sys.path.append(opj(os.path.dirname(os.path.abspath(__file__)), '../dvs_learning/event2cmd/models_seg2cmd'))
# import model as model_library
import importlib.util
from collections import deque

def _strip_module_prefix(sd):
    if not isinstance(sd, dict):
        return sd
    needs_strip = any(k.startswith("module.") for k in sd.keys())
    if not needs_strip:
        return sd
    return {k[len("module."):]: v for k, v in sd.items()}

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

import importlib.util

def _import_sibling_module(module_name, file_path):
    """Import a single .py file by path and register it under module_name."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[module_name] = mod
    return mod

def _load_event2cmd_models(root_event2cmd):
    """
    Ensures 'ViTsubmodules' resolves for BOTH packages, then loads their model.py.
    """
    # --- models_event2seg ---
    e2seg_dir = os.path.join(root_event2cmd, "models_event2seg")
    sys.path.insert(0, e2seg_dir)  # so 'from ViTsubmodules import *' works
    vit_ev_path = os.path.join(e2seg_dir, "ViTsubmodules.py")
    if os.path.isfile(vit_ev_path):
        # Register under the exact name 'ViTsubmodules' so model.py can import it
        _import_sibling_module("ViTsubmodules", vit_ev_path)
    model_ev = _import_sibling_module("model_event2seg", os.path.join(e2seg_dir, "model.py"))

    # --- models_seg2cmd ---
    s2cmd_dir = os.path.join(root_event2cmd, "models_seg2cmd")
    sys.path.insert(0, s2cmd_dir)
    vit_sv_path = os.path.join(s2cmd_dir, "ViTsubmodules.py")
    if os.path.isfile(vit_sv_path):
        # Some repos duplicate this file per package; re-register is harmless
        _import_sibling_module("ViTsubmodules", vit_sv_path)
    model_sv = _import_sibling_module("model_seg2cmd", os.path.join(s2cmd_dir, "model.py"))

    return model_ev, model_sv

class E2EControllerNode:
    def __init__(self):
        self.bridge = CvBridge()

        # Load your trained model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = torch.load("/home/marikan/catkin_ws/src/lane_marking_demo/logs/d08_22_t18_31/nn_controller.pth", map_location=device)
        model_type = "ViT"
        root = "/home/marikan/catkin_ws/src/lane_marking_demo/dvs_learning/event2cmd"
        self.model_event2seg, self.model_seg2cmd = _load_event2cmd_models(root)

        # D: events -> mask
        self.D = getattr(self.model_event2seg, model_type)(output_dim=1, return_sequence=False).to(self.device).float()
        # V: mask -> [steering, velocity]
        self.V = getattr(self.model_seg2cmd, model_type)(output_dim=2, return_sequence=False).to(self.device).float()

        ##### CHANGE HERE EVERY TIME YOU TRAIN A MODEL #####
        state_dict = torch.load("/home/marikan/catkin_ws/src/lane_marking_demo/dvs_learning/event2cmd/training/log/joint_d11_05_t15_05/best_joint_326.pth", map_location=self.device) # ViT, event2cmd
        norm_stats_path = "/home/marikan/catkin_ws/src/lane_marking_demo/dvs_learning/event2cmd/training/log/joint_d11_05_t15_05/norm_stats.txt"

        if not isinstance(state_dict, dict) or ("D_state" not in state_dict and "V_state" not in state_dict):
            rospy.logwarn("Loaded state_dict does not contain 'D_state' or 'V_state' keys; loading entire state_dict.")
            state_D = {}
            state_V = state_dict
        else:
            state_D = state_dict.get("D_state", {})
            state_V = state_dict.get("V_state", {})

        state_D = _strip_module_prefix(state_D)
        state_V = _strip_module_prefix(state_V)
        
        missing_D, unexpected_D = self.D.load_state_dict(state_D, strict=False)
        missing_V, unexpected_V = self.V.load_state_dict(state_V, strict=False)
        if missing_D or unexpected_D:
            print(f"Missing keys when loading D state_dict: {missing_D}")
            print(f"Unexpected keys when loading D state_dict: {unexpected_D}")
        if missing_V or unexpected_V:
            print(f"Missing keys when loading V state_dict: {missing_V}")
            print(f"Unexpected keys when loading V state_dict: {unexpected_V}")

        self.D.eval()
        self.V.eval()

        self.cmd_mean, self.cmd_std = load_norm_stats(norm_stats_path)

        self.pub_raw_pred_mask = rospy.Publisher("/raw_pred_mask", Image, queue_size=1)
        self.pub_binary_pred_mask = rospy.Publisher("/binary_pred_mask", Image, queue_size=1)
        self.pub = rospy.Publisher("/ackermann_drive", AckermannDriveStamped, queue_size=1)
        self.sub = rospy.Subscriber("/dvs_rendering_throttle", Image, self.events_callback)

    def preprocess_events(self, img_msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, "mono8")  # assuming 8-bit grayscale image
        except Exception:
            bgr = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            cv_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        
        tens = torch.from_numpy(cv_img).unsqueeze(0).unsqueeze(0).float() / 255.0  # shape: (1, 1, H, W)
        return tens.to(self.device)

    def preprocess_mask_for_V(self, mask_tensor):
        if mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.unsqueeze(0)  # [1, 1, H, W0]
        if mask_tensor.shape[1] != 1:
            # If D outputs logits or multi-class, you may need argmax / sigmoid
            # Here we assume a 1-channel foreground probability/logit:
            mask_tensor = mask_tensor[:, :1, ...]
        if mask_tensor.min() < 0.0 or mask_tensor.max() > 1.0:
            mask_tensor = torch.sigmoid(mask_tensor)
        if (mask_tensor.shape[2] != 60) or (mask_tensor.shape[3] != 90):
            mask_tensor = torch.nn.functional.interpolate(mask_tensor, size=(60, 90), mode='bilinear', align_corners=False)
        
        # Binarize the mask
        mask_tensor = (mask_tensor > 0.08).float()

        # Publish binary mask for visualization
        mask_np = mask_tensor[0, 0].cpu().numpy()  # assuming single batch, single channel
        mask_np_uint8 = (mask_np * 255).astype(np.uint8)
        msg_binary_mask = self.bridge.cv2_to_imgmsg(mask_np_uint8, "mono8")
        self.pub_binary_pred_mask.publish(msg_binary_mask)
        
        return mask_tensor

    def events_callback(self, msg: Image):
        try:
            with torch.no_grad():
                # events to mask (D)
                ev = self.preprocess_events(msg)
                print(f"[Event2cmd] Received events image of shape: {ev.shape}")
                mask_pred, _ = self.D([ev])
                rospy.loginfo(f"[Event2cmd] ev shape={tuple(ev.shape)}  D_out shape={tuple(mask_pred.shape)}")

                if isinstance(mask_pred, (list, tuple)):
                    mask_pred = mask_pred[0]
                
                # Publish raw predicted mask
                mask_np = mask_pred[0, 0].cpu().numpy()  # assuming single batch, single channel
                mask_np_uint8 = (mask_np * 255).astype(np.uint8)
                msg_raw_mask = self.bridge.cv2_to_imgmsg(mask_np_uint8, "mono8")
                self.pub_raw_pred_mask.publish(msg_raw_mask)

                mask_for_V = self.preprocess_mask_for_V(mask_pred)
                
                # mask to commands (V)
                seq = [mask_for_V]

                cmd_out, _ = self.V(seq)
                if isinstance(cmd_out, (list, tuple)):
                    cmd_out = cmd_out[0]
                steering, velocity = cmd_out[0].cpu().numpy().tolist()  # assuming 2 outputs

                # Denormalize
                steering = steering * self.cmd_std[0] + self.cmd_mean[0]
                velocity = velocity * self.cmd_std[1] + self.cmd_mean[1]
                steering *= 1.0  # scale steering # 5.0 for CNN, 3.0 for LSTMViT

                # Publish command
                msg_cmd = AckermannDriveStamped()
                msg_cmd.drive.steering_angle = float(steering)
                msg_cmd.drive.speed = float(velocity)
                self.pub.publish(msg_cmd)

        except Exception as e:
            rospy.logerr(f"[Event2cmd events callback]: {e}")

def main():
    rospy.init_node("nn_controller_node_event2cmd", anonymous=False)
    node = E2EControllerNode()
    rospy.spin()

if __name__ == "__main__":
    main()