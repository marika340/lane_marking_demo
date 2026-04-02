import os, sys
from os.path import join as opj
import torch
sys.path.append(opj(os.path.dirname(os.path.abspath(__file__)), '../models'))
import model as model_library

# Path to checkpoint
# ckpt_path = "/home/marikan/catkin_ws/src/lane_marking_demo/logs/d08_22_t18_31/best_model_113.pth"
# ckpt_path = "/home/marikan/catkin_ws/src/lane_marking_demo/logs/d08_25_t10_57/model_186.pth"
# ckpt_path = "/home/marikan/catkin_ws/src/lane_marking_demo/logs/d08_27_t15_08/best_model_194.pth"
# ckpt_path = "/home/marikan/catkin_ws/src/lane_marking_demo/logs/d09_01_t22_46/best_model_193.pth"
ckpt_path = "/home/marikan/catkin_ws/src/lane_marking_demo/logs/d09_05_t14_09/best_model_199.pth"

# Recreate model architecture
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_type = "SimpleCNN"
model = getattr(model_library, model_type)(output_dim=2, return_sequence=False).to(device).float()
checkpoint = torch.load(ckpt_path, map_location="cpu")
print(checkpoint.keys())
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)
model.eval()

# Save clean version
torch.save(model.state_dict(), "/home/marikan/catkin_ws/src/lane_marking_demo/logs/d09_05_t14_09/nn_controller.pth")
print("Exported model → nn_controller.pth")
