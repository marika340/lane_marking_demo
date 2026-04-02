#!/usr/bin/env python
import rospy
import math
import tf
import numpy as np

from geometry_msgs.msg import PoseStamped, Pose, Quaternion, Twist
from nav_msgs.msg import Path
from gazebo_msgs.srv import SpawnModel, SpawnModelRequest, GetModelState, SetModelState
from gazebo_msgs.msg import ModelState, ModelStates
from ackermann_msgs.msg import AckermannDriveStamped


class LaneDemoNode(object):
    def __init__(self):
        rospy.init_node("lane_demo_node", anonymous=False)

        # ───────────────────────────────────────────────────────────────────────────
        # 1) Define your waypoints for a continuous figure-8 track (x, y) in world frame
        r = 10.0        # radius of each loop
        n = 500        # total number of sample points around the figure-8
        cx, cy = 0.0, 0.0

        # 2) Lane‐line half‐width (offset from centerline)
        # self.lane_half_width = 0.5  # total lane width = 1.0 m
        self.lane_half_width = 0.2  # total lane width = 0.4 m

        theta = np.linspace(0, 2 * math.pi, n)
        self.waypoints = [
            (cx + r * math.sin(t), cy + r * math.sin(t) * math.cos(t))
            for t in theta
        ]


        # ───────────────────────────────────────────────────────────────────────────

        # 3) Publisher for nav_msgs/Path (for visualization)
        self.path_pub = rospy.Publisher("/planned_path", Path, queue_size=1)

        # 4) Wait for Gazebo services: spawn SDF and get model state
        rospy.loginfo("Waiting for /gazebo/spawn_sdf_model service...")
        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        self.spawn_srv = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)

        rospy.loginfo("Waiting for /gazebo/get_model_state service...")
        rospy.wait_for_service("/gazebo/get_model_state")
        self.get_state_srv = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)

        rospy.loginfo("Waiting for /gazebo/set_model_state service...")
        rospy.wait_for_service("/gazebo/set_model_state")
        self.set_state_srv = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        self.camera_model_name = "camera_model"  # must match the <model name> in SDF

        rospy.sleep(2.0)  # wait a bit for Gazebo startup
        pose = self.get_camera_pose()
        if pose is None:
            rospy.logwarn("Camera model not found in Gazebo!")
        else:
            rospy.loginfo(f"Camera model found at {pose}")
            
        # 5) Spawn lane‐line boxes
        self.spawn_lane_markings()

        # 6) Publish the Path (just once)
        self.publish_path()

        # 7) Prepare for physics‐based motion
        self.planned_path = self.current_path.poses
        self.time_index = 0
        # self.camera_model_name = "camera_model"  # must match the <model name> in SDF

        # 7a) Publisher for velocity commands
        #     This topic name (cmd_vel) must match the plugin remapping in your SDF.
        self.cmd_vel_pub = rospy.Publisher("/lane_demo/cmd_vel", Twist, queue_size=1)

        # 7b) Publisher for AckermannDrive
        self.ackermann_pub = rospy.Publisher("/lane_demo/ackermann_cmd", AckermannDriveStamped, queue_size=1)


        # 8) Timer: run controller at 20 Hz
        self.controller_timer = rospy.Timer(rospy.Duration(0.05), self.path_follower_callback)

        rospy.loginfo("LaneDemoNode initialized (using physics‐based cmd_vel control).")

    # ─────────────────────────────────────────────────────────────────────────────
    def spawn_lane_markings(self):
        """
        For each consecutive waypoint segment, spawn two thin white boxes (lane lines)
        offset left/right by lane_half_width. Each box is 0.01 m high, so we must place
        its center at z = 0.005 to be flush with the asphalt (z = 0).
        """
        listener = tf.TransformListener()  # only used for quaternion calculations

        for i in range(len(self.waypoints) - 1):
            x0, y0 = self.waypoints[i]
            x1, y1 = self.waypoints[i + 1]

            # 1) Compute segment heading and length
            dx = x1 - x0
            dy = y1 - y0
            segment_length = math.hypot(dx, dy)
            heading = math.atan2(dy, dx)

            # 2) Midpoint of this segment
            mid_x = (x0 + x1) / 2.0
            mid_y = (y0 + y1) / 2.0

            # 3) Normal vector to offset left/right
            nx = -math.sin(heading)
            ny = math.cos(heading)

            # 4) Offset centers
            left_center_x = mid_x + nx * self.lane_half_width
            left_center_y = mid_y + ny * self.lane_half_width
            right_center_x = mid_x - nx * self.lane_half_width
            right_center_y = mid_y - ny * self.lane_half_width

            # 5) Box dimensions (x‐axis = length, y‐axis = 0.1 m width, z‐axis = 0.01 m height)
            sx = segment_length
            sy = 0.1
            sz = 0.01

            # 6) Color: white
            r, g, b, a = 1.0, 1.0, 1.0, 1.0

            box_sdf_template = """
            <?xml version="1.0" ?>
            <sdf version="1.6">
              <model name="{model_name}">
                <static>true</static>
                <link name="link">
                  <!-- center at z = sz/2 so top surface is flush with z=0 -->
                  <pose>0 0 {half_z:.4f} 0 0 0</pose>
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

            # 7) Spawn “left” and “right” boxes
            for side, (cx, cy) in [
                ("left", (left_center_x, left_center_y)),
                ("right", (right_center_x, right_center_y))
            ]:
                # Skip lane boxes that are too close to the origin (intersection)
                if math.hypot(mid_x, mid_y) < self.lane_half_width:
                    continue

                model_name = "lane_line_{}_{}".format(side, i)
                half_z = sz / 2.0
                box_sdf = box_sdf_template.format(
                    model_name=model_name,
                    half_z=half_z,
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
                req.initial_pose.position.z = 0.0  # link’s own pose centers the box
                # Rotate to align with segment heading
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


    # ─────────────────────────────────────────────────────────────────────────────
    def publish_path(self):
        """
        Publish a nav_msgs/Path message that goes through each waypoint with orientation.
        """
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "map"

        # Precompute headings between consecutive waypoints
        headings = []
        for i in range(len(self.waypoints) - 1):
            x0, y0 = self.waypoints[i]
            x1, y1 = self.waypoints[i + 1]
            headings.append(math.atan2(y1 - y0, x1 - x0))
        headings.append(headings[-1])  # reuse final heading

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

    # ─────────────────────────────────────────────────────────────────────────────
    def get_camera_pose(self):
        """
        Query Gazebo for current pose of camera_model.
        Returns: (x, y, yaw) in the ‘map’ (world) frame.
        """
        try:
            res = self.get_state_srv(self.camera_model_name, "world")
            if not res.success:
                rospy.logwarn("GetModelState failed: {}".format(res.status_message))
                return None
            px = res.pose.position.x
            py = res.pose.position.y
            # Extract yaw from quaternion
            q = res.pose.orientation
            _, _, yaw = tf.transformations.euler_from_quaternion([
                q.x, q.y, q.z, q.w
            ])
            return (px, py, yaw)
        except rospy.ServiceException as e:
            rospy.logerr("get_model_state service call failed: {}".format(e))
            return None

    # ─────────────────────────────────────────────────────────────────────────────
    def path_follower_callback(self, event):
        """
        Every 0.05s (20Hz), read current camera pose from Gazebo, compute a Twist towards
        the next waypoint, and publish it on /lane_demo/cmd_vel. When close enough, advance index.
        """
        if not hasattr(self, "planned_path") or len(self.planned_path) == 0:
            return

        # 1) Get current camera pose (x, y, yaw)
        cam_pose = self.get_camera_pose()
        if cam_pose is None:
            return
        cam_x, cam_y, cam_yaw = cam_pose

        # 2) Determine target waypoint
        if self.time_index >= len(self.planned_path):
            self.time_index = 0  # loop back
        target = self.planned_path[self.time_index].pose
        tx = target.position.x
        ty = target.position.y

        # 3) Compute error in position
        dx = tx - cam_x
        dy = ty - cam_y
        dist = math.hypot(dx, dy)

         # 4) Compute the desired heading and steering error
        desired_yaw = math.atan2(dy, dx)
        yaw_error = self.normalize_angle(desired_yaw - cam_yaw)

        # 5) Proportional control
        k_steer = 1.5  # Gain for steering angle
        k_speed = 1.0  # Gain for speed

        steering_angle = max(-0.418, min(k_steer * yaw_error, 0.418))  # limit to ±24 degrees
        speed = max(0.0, min(k_speed * dist, 2.0))  # limit to 2.0 m/s

        # 6) Publish AckermannDriveStamped
        ackermann_msg = AckermannDriveStamped()
        ackermann_msg.header.stamp = rospy.Time.now()
        ackermann_msg.drive.steering_angle = steering_angle
        ackermann_msg.drive.speed = speed

        self.ackermann_pub.publish(ackermann_msg)

        # # 4) Simple Proportional Control Gains
        # k_lin = 0.8    # linear speed gain
        # k_ang = 4.0    # angular speed gain

        # # 5) Compute desired heading
        # desired_yaw = math.atan2(dy, dx)
        # yaw_error = self.normalize_angle(desired_yaw - cam_yaw)

        # # 6) Build and publish Twist
        # twist = Twist()
        # # If we’re far from waypoint, drive forward; otherwise, zero out
        # # twist.linear.x = max(0.0, min(k_lin * dist, 1.0))  # clip between 0 and 1.0 m/s
        # # twist.angular.z = max(-2.0, min(k_ang * yaw_error, 2.0))  # clip ±2 rad/s
        # twist.linear.x = k_lin * dist
        # twist.angular.z = k_ang * yaw_error
        # self.cmd_vel_pub.publish(twist)

        state_msg = ModelState()
        state_msg.model_name = self.camera_model_name
        state_msg.pose = Pose()
        state_msg.pose.position = target.position
        # Copy over the orientation we stored in 'publish_path()'
        state_msg.pose.orientation = target.orientation

        # Zero out linear/angular velocity
        state_msg.twist.linear.x = 0.0
        state_msg.twist.linear.y = 0.0
        state_msg.twist.linear.z = 0.0
        state_msg.twist.angular.x = 0.0
        state_msg.twist.angular.y = 0.0
        state_msg.twist.angular.z = 0.0

        state_msg.reference_frame = "world"

        try:
            res = self.set_state_srv(state_msg)
            if not res.success:
                rospy.logwarn("set_model_state failed: {}".format(res.status_message))
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: {}".format(e))

        # 7) If we’re close enough, advance to the next waypoint
        # if dist < 0.2:
        self.time_index += 1
        if self.time_index >= len(self.planned_path):
            self.time_index = 0

    # ─────────────────────────────────────────────────────────────────────────────
    @staticmethod
    def normalize_angle(angle):
        """
        Normalize an angle difference to the range [-pi, +pi].
        """
        return math.atan2(math.sin(angle), math.cos(angle))

# ─────────────────────────────────────────────────────────────────────────────────
def main():
    try:
        node = LaneDemoNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()
