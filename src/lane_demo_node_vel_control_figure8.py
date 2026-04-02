#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
import tf
import numpy as np

from geometry_msgs.msg import PoseStamped, Pose, Quaternion, Twist
from nav_msgs.msg import Path
from gazebo_msgs.srv import SpawnModel, SpawnModelRequest, GetModelState, SetModelState
from gazebo_msgs.msg import ModelState, ModelStates
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool

class LaneDemoNode(object):
    def __init__(self):
        rospy.init_node("lane_demo_node_vel_control_figure8", anonymous=False)

        self.prev_speed = 0.0
        self.prev_acceleration = 0.0
        self.prev_steering_angle = 0.0
        self.prev_time = rospy.Time.now()

        # 1) Define your waypoints for a continuous figure-8 track (x, y) in world frame
        r = 10.0        # radius of each loop
        n = 500        # total number of sample points around the figure-8
        cx, cy = 0.0, 0.0

        # 2) Lane-line half-width (offset from centerline)
        self.lane_half_width = 0.2  # total lane width = 1.0 m
        theta = np.linspace(0, 2 * math.pi, n)
        # self.waypoints = [
        #     (cx + r * math.sin(t), cy + r * math.sin(t) * math.cos(t))
        #     for t in theta
        # ]
        a = r  # semi-major axis
        b = r / 2  # semi-minor axis for smoother vertical bend
        self.waypoints = [
            (a * math.sin(t), b * math.sin(2 * t))
            for t in np.linspace(0, 2 * math.pi, n)
        ]
        np.save("/home/marikan/catkin_ws/src/lane_marking_demo/worlds/track_figure8.npy", self.waypoints)

        # 3) Publisher for nav_msgs/Path (for visualization)
        self.path_pub = rospy.Publisher("/planned_path", Path, queue_size=1)

        # 3.5) Publisher for AckermannDriveStamped (for creating dataset)
        self.ackermann_pub = rospy.Publisher("/ackermann_drive", AckermannDriveStamped, queue_size=1)
        self.ready_pub = rospy.Publisher("/lane_demo/ready", Bool, queue_size=1)

        # 4) Wait for Gazebo services: spawn SDF and get model state
        rospy.loginfo("Waiting for /gazebo/spawn_sdf_model service...")
        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        self.spawn_srv = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)

        rospy.loginfo("Waiting for /gazebo/get_model_state service...")
        rospy.wait_for_service("/gazebo/get_model_state")
        self.get_state_srv = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)

        self.camera_model_name = "camera_model"  # must match the <model name> in SDF

        rospy.sleep(2.0)  # wait a bit for Gazebo startup
        pose = self.get_camera_pose()
        if pose is None:
            rospy.logwarn("Camera model not found in Gazebo!")
        else:
            rospy.loginfo(f"Camera model found at {pose}")
            
        # 5) Spawn lane-line boxes
        self.spawn_lane_markings()

        # 6) Publish the Path (just once)
        self.publish_path()

        # 7) Prepare for physics-based motion
        self.planned_path = self.current_path.poses
        self.time_index = 0

        # 7a) Publisher for velocity commands
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        self.ready_pub.publish(Bool(data=True))

        # 8) Timer: run controller at 20 Hz
        self.controller_timer = rospy.Timer(rospy.Duration(0.05), self.path_follower_callback)

        rospy.loginfo("LaneDemoNode initialized (using physics-based cmd_vel control).")

    def spawn_lane_markings(self):
        """
        For each consecutive waypoint segment, spawn two thin white boxes (lane lines)
        offset left/right by lane_half_width. Each box is 0.01 m high, so we must place
        its center at z = 0.005 to be flush with the asphalt (z = 0).
        """
        listener = tf.TransformListener()

        for i in range(len(self.waypoints) - 1):
            x0, y0 = self.waypoints[i]
            x1, y1 = self.waypoints[i + 1]

            dx = x1 - x0
            dy = y1 - y0
            segment_length = math.hypot(dx, dy)
            heading = math.atan2(dy, dx)

            mid_x = (x0 + x1) / 2.0
            mid_y = (y0 + y1) / 2.0

            nx = -math.sin(heading)
            ny = math.cos(heading)

            left_center_x = mid_x + nx * self.lane_half_width
            left_center_y = mid_y + ny * self.lane_half_width
            right_center_x = mid_x - nx * self.lane_half_width
            right_center_y = mid_y - ny * self.lane_half_width

            sx = segment_length + 0.02 # Added 0.02 so that the outer lane has overlap
            sy = 0.01
            sz = 0.01

            r, g, b, a = 1.0, 1.0, 1.0, 1.0

            box_sdf_template = """
            <?xml version="1.0" ?>
            <sdf version="1.6">
              <model name="{model_name}">
                <static>true</static>
                <link name="link">
                  <pose>0 0 {top_z:.4f} 0 0 0</pose>
                  <visual name="visual">
                    <geometry>
                      <box>
                        <size>{sx:.4f} {sy:.4f} {sz:.4f}</size>
                      </box>
                    </geometry>
                    <material>
                      <ambient>{r:.2f} {g:.2f} {b:.2f} {a:.2f}</ambient>
                      <diffuse>{r:.2f} {g:.2f} {b:.2f} {a:.2f}</diffuse>
                    </material>
                  </visual>
                  <collision name="collision">
                    <geometry>
                      <box>
                        <size>{sx:.4f} {sy:.4f} {sz:.4f}</size>
                      </box>
                    </geometry>
                  </collision>
                </link>
              </model>
            </sdf>
            """

            for side, (cx, cy) in [
                ("left", (left_center_x, left_center_y)),
                ("right", (right_center_x, right_center_y))
            ]:
                if math.hypot(mid_x, mid_y) < self.lane_half_width:
                    continue

                model_name = "lane_line_{}_{}".format(side, i)
                top_z = -sz / 2.0 + 0.001
                box_sdf = box_sdf_template.format(
                    model_name=model_name,
                    top_z=top_z,
                    sx=sx, sy=sy, sz=sz,
                    r=r, g=g, b=b, a=a
                )

                req = SpawnModelRequest()
                req.model_name = model_name
                req.model_xml = box_sdf
                req.robot_namespace = ""
                req.reference_frame = "world"
                req.initial_pose.position.x = cx
                req.initial_pose.position.y = cy
                req.initial_pose.position.z = 0.0
                q = tf.transformations.quaternion_from_euler(0, 0, heading)
                req.initial_pose.orientation.x = q[0]
                req.initial_pose.orientation.y = q[1]
                req.initial_pose.orientation.z = q[2]
                req.initial_pose.orientation.w = q[3]

                try:
                    res = self.spawn_srv(req)
                    if res.success:
                        rospy.loginfo("Spawned lane_line: {}".format(model_name))
                    else:
                        rospy.logwarn("Failed to spawn {}: {}".format(model_name, res.status_message))
                except rospy.ServiceException as e:
                    rospy.logerr("spawn_sdf_model service call failed: {}".format(e))

    def publish_path(self):
        """
        Publish a nav_msgs/Path message that goes through each waypoint with orientation.
        """
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "map"

        headings = []
        for i in range(len(self.waypoints) - 1):
            x0, y0 = self.waypoints[i]
            x1, y1 = self.waypoints[i + 1]
            headings.append(math.atan2(y1 - y0, x1 - x0))
        headings.append(headings[-1])

        for i, (x, y) in enumerate(self.waypoints):
            ps = PoseStamped()
            ps.header.stamp = rospy.Time.now()
            ps.header.frame_id = "map"
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.position.z = 0.0

            yaw = headings[i]
            q = tf.transformations.quaternion_from_euler(0, 0, yaw)
            ps.pose.orientation = Quaternion(q[0], q[1], q[2], q[3])
            path_msg.poses.append(ps)

        self.current_path = path_msg
        self.path_pub.publish(path_msg)
        rospy.loginfo("Published Path ({} waypoints)".format(len(self.waypoints)))

    def get_camera_pose(self):
        """
        Query Gazebo for current pose of camera_model.
        Returns: (x, y, yaw) in the 'map' (world) frame.
        """
        try:
            res = self.get_state_srv(self.camera_model_name, "world")
            if not res.success:
                rospy.logwarn("GetModelState failed: {}".format(res.status_message))
                return None
            px = res.pose.position.x
            py = res.pose.position.y
            q = res.pose.orientation
            _, _, yaw = tf.transformations.euler_from_quaternion([
                q.x, q.y, q.z, q.w
            ])
            return (px, py, yaw)
        except rospy.ServiceException as e:
            rospy.logerr("get_model_state service call failed: {}".format(e))
            return None

    def path_follower_callback(self, event):
        """
        Every 0.05s (20Hz), read current camera pose from Gazebo, compute a Twist towards
        the next waypoint, and publish it on /lane_demo/cmd_vel. When close enough, advance index.
        """
        if not hasattr(self, "planned_path") or len(self.planned_path) == 0:
            return

        cam_pose = self.get_camera_pose()
        if cam_pose is None:
            return
        cam_x, cam_y, cam_yaw = cam_pose

        # if self.time_index >= len(self.planned_path):
        #     self.time_index = 0
        # target = self.planned_path[self.time_index].pose

        # Parameters
        best_index = None
        best_dist = float('inf')
        min_lookahead_distance = 1.0
        cam_pose_vec = np.array([cam_x, cam_y])

        # Only search a limited window ahead of current index
        search_window = 100
        start_index = self.time_index
        end_index = min(start_index + search_window, len(self.planned_path))

        # Precompute cos/sin of yaw for rotation
        cos_yaw = math.cos(-cam_yaw)
        sin_yaw = math.sin(-cam_yaw)

        for i in range(start_index, end_index):
        # for i in range(len(self.planned_path)):
            wp = self.planned_path[i].pose
            dx = wp.position.x - cam_x
            dy = wp.position.y - cam_y

            # Transform waypoint to vehicle frame
            x_rel = dx * cos_yaw - dy * sin_yaw
            y_rel = dx * sin_yaw + dy * cos_yaw

            if x_rel > 0.1:  # in front of vehicle
                dist = math.hypot(x_rel, y_rel)
                if dist > min_lookahead_distance and dist < best_dist:
                    best_dist = dist
                    best_index = i


        # Fallback: just go to closest global waypoint if nothing ahead
        if best_index is not None:
            self.time_index = best_index

        else:
            # fallback: just keep going forward slowly
            self.time_index = min(self.time_index + 1, len(self.planned_path) - 1)
            # self.time_index = np.argmin([
            #     math.hypot(wp.pose.position.x - cam_x, wp.pose.position.y - cam_y)
            #     for wp in self.planned_path
            # ])
            # fallback that also checks x_rel > 0.1
            # self.time_index = np.argmin([
            #     math.hypot(wp.pose.position.x - cam_x, wp.pose.position.y - cam_y)
            #     if (wp.pose.position.x - cam_x) * math.cos(cam_yaw) +
            #     (wp.pose.position.y - cam_y) * math.sin(cam_yaw) > 0  # dot product > 0
            #     else float('inf')
            #     for wp in self.planned_path
            # ])

        target = self.planned_path[self.time_index].pose

        tx = target.position.x
        ty = target.position.y

        dx = tx - cam_x
        dy = ty - cam_y
        dist = math.hypot(dx, dy)

        k_lin = 6.0
        k_ang = 6.0

        desired_yaw = math.atan2(dy, dx)
        yaw_error = self.normalize_angle(desired_yaw - cam_yaw)

        twist = Twist()
        progress = self.time_index / float(len(self.planned_path))
        speed_scale = 0.5 * math.sin(2 * math.pi * progress * 2) + 1.0
        # rospy.loginfo("Speed scale: {}".format(speed_scale))
        # rospy.loginfo("k_lin * dist: {}".format(k_lin * dist))
        # if (k_lin * dist) > speed_scale:
        #     rospy.loginfo("Speed scale is lower, limiting to {}".format(speed_scale))
        twist.linear.x = min(k_lin * dist, speed_scale)
        # twist.linear.x = k_lin * dist
        twist.angular.y = 0.0
        twist.angular.z = k_ang * yaw_error
        self.cmd_vel_pub.publish(twist)

        # Get current values
        current_time = rospy.Time.now()
        dt = (current_time - self.prev_time).to_sec()
        dt = max(dt, 1e-6)  # prevent division by zero

        current_speed = twist.linear.x
        current_angle = twist.angular.z

        # Compute derivatives
        acceleration = (current_speed - self.prev_speed) / dt
        jerk = (acceleration - self.prev_acceleration) / dt
        steering_angle_velocity = (current_angle - self.prev_steering_angle) / dt

        # Create message
        ackermann_msg = AckermannDriveStamped()
        ackermann_msg.header.stamp = current_time
        ackermann_msg.header.frame_id = "base_link"

        ackermann_msg.drive.speed = current_speed
        ackermann_msg.drive.steering_angle = current_angle
        ackermann_msg.drive.acceleration = acceleration
        ackermann_msg.drive.jerk = jerk
        ackermann_msg.drive.steering_angle_velocity = steering_angle_velocity

        # Publish
        self.ackermann_pub.publish(ackermann_msg)

        # Store previous values
        self.prev_time = current_time
        self.prev_speed = current_speed
        self.prev_acceleration = acceleration
        self.prev_steering_angle = current_angle
        
        if dist < 0.3:
            self.time_index = (self.time_index + 1) % len(self.planned_path)


    @staticmethod
    def normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

def main():
    try:
        node = LaneDemoNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()
