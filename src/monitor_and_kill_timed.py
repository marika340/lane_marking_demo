#!/usr/bin/env python3
import rospy
import argparse
import time
import math
from gazebo_msgs.msg import ModelStates
import os
import signal

class MonitorAndKill:
    def __init__(self, track, duration, depart_thresh=5.0, tolerance=0.5):
        self.track = track
        self.duration = duration
        self.tolerance = tolerance
        self.start_time = time.time()
        self.initial_pose = None
        self.depart_threshold = depart_thresh
        self.has_left_start = False
        self.model_name = "camera_model"  # adjust if your vehicle name differs
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.state_cb)

    def state_cb(self, msg):
        try:
            idx = msg.name.index(self.model_name)
        except ValueError:
            return  # vehicle not spawned yet

        pose = msg.pose[idx]
        x, y = pose.position.x, pose.position.y

        if self.initial_pose is None:
            self.initial_pose = (x, y)
            rospy.loginfo(f"[Monitor] Initial pose recorded: {self.initial_pose}")
            return

        # Check distance to starting point
        dx = x - self.initial_pose[0]
        dy = y - self.initial_pose[1]
        dist = math.sqrt(dx*dx + dy*dy)

        elapsed = time.time() - self.start_time

        if not self.has_left_start:
            if dist > self.depart_threshold:
                self.has_left_start = True
                rospy.loginfo(f"[Monitor] Vehicle has left the start area.")
            return
        rospy.loginfo_throttle(0.5, f"Distance to start: {dist:.2f} | Has left: {self.has_left_start}")
        if dist < self.tolerance:
            rospy.loginfo(f"[Monitor] Car returned to starting point (dist={dist:.2f}). Ending run.")
            self.shutdown()
        elif elapsed >= self.duration:
            rospy.loginfo(f"[Monitor] Duration {self.duration}s reached. Ending run.")
            self.shutdown()

    def shutdown(self):
        # Stop ROS node
        rospy.signal_shutdown("Condition met")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", required=True, help="Track name")
    parser.add_argument("--duration", type=int, required=True, help="Max run duration (seconds)")
    parser.add_argument("--depart_thresh", type=float, default=5.0, help="Meters to confirm lap started")
    parser.add_argument("--tolerance", type=float, default=0.5, help="Distance threshold to stop early")
    args = parser.parse_args()

    rospy.init_node("monitor_and_kill_timed", anonymous=True)
    monitor = MonitorAndKill(track=args.track, duration=args.duration, depart_thresh=args.depart_thresh, tolerance=args.tolerance)
    rospy.spin()
