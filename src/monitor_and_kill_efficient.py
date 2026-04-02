#!/usr/bin/env python
import rospy
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Bool
import time
import argparse
import os

TRACK_NODE_MAP = {
    "circle": "/lane_demo_node_vel_control_circle",
    "figure8": "/lane_demo_node_vel_control_figure8_rand_start",
    "figure8_opposite": "/lane_demo_node_vel_control_figure8_opposite",
    "square_smooth": "/lane_demo_node_vel_control_square_smooth",
    "square_random": "/lane_demo_node_vel_control_square_random",
}

class MonitorReturn:
    def __init__(self, track):
        self.initial_pose = None
        self.latest_pose = None
        self.ready = False
        self.has_left_start = False
        self.shutdown_triggered = False
        self.threshold = 1.0        # meters to consider "returned"
        self.depart_threshold = 5.0 # meters to confirm lap started
        self.vehicle_name = "camera_model"
        self.track = track
        self.node_to_kill = TRACK_NODE_MAP.get(track, None)

        rospy.init_node("monitor_and_kill_efficient")

        rospy.Subscriber("/lane_demo/ready", Bool, self.ready_callback)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.pose_callback)

    def ready_callback(self, msg):
        self.ready = msg.data
        rospy.loginfo("Received /lane_demo/ready == True")

    def pose_callback(self, msg):
        if not self.ready or self.shutdown_triggered:
            return

        if self.vehicle_name not in msg.name:
            return

        idx = msg.name.index(self.vehicle_name)
        self.latest_pose = msg.pose[idx]
        pos = self.latest_pose.position

        if self.initial_pose is None:
            self.initial_pose = pos
            rospy.loginfo(f"Initial position recorded: {self.initial_pose}")
            return

        dx = pos.x - self.initial_pose.x
        dy = pos.y - self.initial_pose.y
        dist = (dx ** 2 + dy ** 2) ** 0.5

        rospy.loginfo_throttle(0.5, f"Distance to start: {dist:.2f} | Has left: {self.has_left_start}")

        # Confirm lap start
        if not self.has_left_start:
            if dist > self.depart_threshold:
                self.has_left_start = True
                rospy.loginfo("Vehicle has left the start area.")
            return

        # Detect lap completion
        if dist < self.threshold:
            rospy.loginfo("Vehicle completed a lap. Killing trial processes...")
            self.shutdown_triggered = True

            time.sleep(5)  # allow dataset node to flush

            # Kill trial-specific nodes safely
            if self.node_to_kill:
                os.system(f"rosnode kill {self.node_to_kill}")
            os.system("rosnode kill /lane_segmentation_dataset_node")
            os.system("rosnode kill /dvs_renderer")

            rospy.loginfo("Monitor finished, shutting itself down...")
            time.sleep(1)
            rospy.signal_shutdown("Trial finished")

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", type=str, default="figure8",
                        choices=["circle", "figure8", "figure8_opposite", "square_smooth", "square_random"])
    parser.add_argument("--duration_min", type=float, default = 5.0)
    parser.add_argument("--duration_max", type=float, default = 15.0)
    args = parser.parse_args()

    duration_min = args.duration_min
    duration_max = args.duration_max
    duration = duration_min + (duration_max - duration_min) * np.random.rand()
    rospy.loginfo(f"Trial duration set to {duration:.1f} seconds")
    rospy.Timer(rospy.Duration(duration), lambda event: rospy.signal_shutdown("Time limit reached"), oneshot=True)

    monitor = MonitorReturn(args.track)
    monitor.run()
