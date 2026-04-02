import cv2
import numpy as np
import os

def generate_ground_truth_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    kernel = np.ones((1, 1), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

# Paths
input_dir = os.path.expanduser("~/catkin_ws/src/lane_marking_demo/dataset101/rgb")
output_dir = os.path.expanduser("~/catkin_ws/src/lane_marking_demo/dataset101/masks1")
os.makedirs(output_dir, exist_ok=True)

# Process each image
for filename in sorted(os.listdir(input_dir)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to load {filename}")
            continue

        mask = generate_ground_truth_mask(image)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, mask)

print("Mask generation complete.")
