#!/usr/bin/env python
import rospy
import cv2
import os
import shutil
from cv_bridge import CvBridge
import numpy as np
from sensor_msgs.msg import Image
from gazebo_msgs.msg import ModelStates
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool
from datetime import datetime
import rospkg
import csv
import threading
import queue

class LaneSegmentationDatasetCollector:
    def __init__(self):
        rospy.init_node("lane_segmentation_dataset_node_2")

        # Output directory
        self.output_dir = os.path.expanduser("~/catkin_ws/src/lane_marking_demo/dataset")
        os.makedirs(os.path.join(self.output_dir, "events"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "masks"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "metadata_files"), exist_ok=True)

        self.metadata_path = os.path.join(self.output_dir, "metadata.txt")
        self.csv_path = os.path.join(self.output_dir, "control_odometry.csv")
        self.init_csv_file()


        # CV Bridge
        self.bridge = CvBridge()
        self.latest_cmd = None
        self.latest_pose = None
        self.ready = False

        # RGB saving thread
        self.rgb_queue = queue.Queue()
        self.rgb_thread = threading.Thread(target=self.rgb_saver_loop, daemon=True)
        self.rgb_thread.start()

        # Subscribers
        rospy.Subscriber("/dvs_rendering_throttle", Image, self.dvs_callback)
        rospy.Subscriber("/camera/rgb_camera/rgb/image_raw", Image, self.rgb_callback, queue_size=100)
        rospy.Subscriber("/ackermann_drive", AckermannDriveStamped, self.ackermann_callback)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.model_states_callback)
        rospy.Subscriber("/lane_demo/ready", Bool, self.ready_callback)

        # Save metadata and reference files once
        self.save_metadata()
        self.copy_reference_files()

    def init_csv_file(self):
        with open(self.csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "timestamp",
                "steering_angle",
                "steering_angle_velocity",
                "speed",
                "acceleration",
                "jerk",
                "position_x",
                "position_y",
                "position_z",
                "orientation_x",
                "orientation_y",
                "orientation_z",
                "orientation_w"
            ])


    def ready_callback(self, msg):
        self.ready = msg.data

    def ackermann_callback(self, msg):
        self.latest_cmd = msg.drive

    def model_states_callback(self, msg):
        if "camera_model" in msg.name:
            idx = msg.name.index("camera_model")
            self.latest_pose = msg.pose[idx]

    def dvs_callback(self, msg):
        if not self.ready or self.latest_cmd is None or self.latest_pose is None:
            return

        try:
            dvs_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"Error converting DVS image: {e}")
            return

        timestamp = rospy.Time.now().to_nsec()
        cv2.imwrite(os.path.join(self.output_dir, "events", f"{timestamp}.png"), dvs_image)

        # Append full AckermannDriveStamped + pose
        with open(self.csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            pos = self.latest_pose.position
            ori = self.latest_pose.orientation
            drive = self.latest_cmd
            writer.writerow([
                timestamp,
                drive.steering_angle,
                drive.steering_angle_velocity,
                drive.speed,
                drive.acceleration,
                drive.jerk,
                pos.x, pos.y, pos.z,
                ori.x, ori.y, ori.z, ori.w
            ])


        rospy.loginfo(f"Saved DVS, command, odometry at {timestamp}")

    # def rgb_callback(self, msg):
    #     if not self.ready:
    #         return

    #     try:
    #         rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    #     except Exception as e:
    #         rospy.logerr(f"Error converting RGB image: {e}")
    #         return

    #     mask = self.generate_ground_truth_mask(rgb_image)

    #     timestamp = rospy.Time.now().to_nsec()
    #     cv2.imwrite(os.path.join(self.output_dir, "rgb", f"{timestamp}.png"), rgb_image)
    #     cv2.imwrite(os.path.join(self.output_dir, "masks", f"{timestamp}.png"), mask)

    #     rospy.loginfo(f"Saved RGB + mask at {timestamp}")

    def rgb_callback(self, msg):
        if not self.ready:
            return

        try:
            rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"Error converting RGB image: {e}")
            return

        timestamp = msg.header.stamp.to_nsec()
        # mask = self.generate_ground_truth_mask(rgb_image)

        # self.rgb_queue.put((timestamp, rgb_image, mask))
        self.rgb_queue.put((timestamp, rgb_image))
        rospy.loginfo(f"[RGB QUEUE] Enqueued frame at {timestamp}")

    def rgb_saver_loop(self):
        while not rospy.is_shutdown():
            try:
                timestamp, rgb_image, mask = self.rgb_queue.get(timeout=1.0)
                cv2.imwrite(os.path.join(self.output_dir, "rgb", f"{timestamp}.png"), rgb_image)
                cv2.imwrite(os.path.join(self.output_dir, "masks", f"{timestamp}.png"), mask)
                rospy.loginfo(f"[RGB SAVED] {timestamp}")
            except queue.Empty:
                continue
            
    def generate_ground_truth_mask(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def save_metadata(self):
        # Wait until the vehicle pose is received
        rospy.loginfo("Waiting for initial camera_model pose...")
        while self.latest_pose is None and not rospy.is_shutdown():
            rospy.sleep(0.1)

        # Get x, y, z
        pos = self.latest_pose.position
        x, y, z = pos.x, pos.y, pos.z

        # Convert quaternion to RPY (roll, pitch, yaw)
        import tf
        ori = self.latest_pose.orientation
        quat = (ori.x, ori.y, ori.z, ori.w)
        roll, pitch, yaw = tf.transformations.euler_from_quaternion(quat)

        # This is an f-string — no .format() needed
        metadata = f"""
Dataset Metadata
----------------
trajectory_type: [circle | figure8 | tight_square | smooth_square]
camera_height: see metadata_files/camera_model.sdf <pose>0 0 0.1 0 0 0</pose>
initial_vehicle_position:
  <pose>0 0 0.2 0 0 0</pose>
  x: {x:.3f}
  y: {y:.3f}
  z: {z:.3f}
  R (roll): {roll:.3f}
  P (pitch): {pitch:.3f}
  Y (yaw): {yaw:.3f}
event_threshold: 2
background_world: metadata_files/random_trial.world
k_lin: 6.0
k_ang: 6.0
distance_threshold: 0.3

This dataset was collected using ROS on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""".strip()

        with open(self.metadata_path, "w") as f:
            f.write(metadata)

        rospy.loginfo("Saved metadata with initial pose.")


    def copy_reference_files(self):
        meta_dir = os.path.join(self.output_dir, "metadata_files")

        # Copy camera_model.sdf
        src_camera = os.path.expanduser("~/catkin_ws/src/lane_marking_demo/worlds/camera_model.sdf")
        dst_camera = os.path.join(meta_dir, "camera_model.sdf")
        try:
            shutil.copy(src_camera, dst_camera)
            rospy.loginfo("Copied camera_model.sdf to metadata_files/")
        except Exception as e:
            rospy.logwarn(f"Could not copy camera_model.sdf: {e}")

        # Copy random_trial.world using rospkg
        try:
            rospack = rospkg.RosPack()
            lane_demo_path = rospack.get_path("lane_marking_demo")
            src_world = os.path.join(lane_demo_path, "worlds", "random_trial.world")
            dst_world = os.path.join(meta_dir, "random_trial.world")
            shutil.copy(src_world, dst_world)
            rospy.loginfo("Copied random_trial.world to metadata_files/")
        except Exception as e:
            rospy.logwarn(f"Could not copy random_trial.world: {e}")

if __name__ == '__main__':
    try:
        node = LaneSegmentationDatasetCollector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
