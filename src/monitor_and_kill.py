#!/usr/bin/env python
import rospy
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Bool
import os
import time

class MonitorReturn:
    def __init__(self):
        self.initial_pose = None
        self.latest_pose = None
        self.ready = False
        self.has_left_start = False
        self.shutdown_triggered = False
        self.threshold = 1.0  # meters to return
        self.depart_threshold = 5.0  # meters to confirm lap started
        self.vehicle_name = "camera_model"

        rospy.init_node("monitor_and_kill")

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

        if not self.has_left_start:
            if dist > self.depart_threshold:
                self.has_left_start = True
                rospy.loginfo("Vehicle has left the start area.")
            return

        if dist < self.threshold:
            rospy.loginfo("Vehicle completed a lap. Killing background processes.")
            self.shutdown_triggered = True

            time.sleep(5)  # Let dataset node flush

            os.system("pkill -f lane_demo_node_vel_control_circle.py")
            os.system("pkill -f spawn_random_pose.py")
            os.system("pkill -f dvs_renderer")
            os.system("pkill -f gzclient")
            os.system("pkill -f gzserver")

            # Kill all other nodes except monitor_and_kill
            os.system("rosnode list | grep -v monitor_and_kill | xargs -r rosnode kill")

            rospy.loginfo("Shutting down monitor_and_kill.py...")
            time.sleep(1)
            rospy.signal_shutdown("Done")


    def run(self):
        rospy.spin()

if __name__ == "__main__":
    MonitorReturn().run()
