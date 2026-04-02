#!/usr/bin/env python
import rospy
import math
import tf
from geometry_msgs.msg import PoseStamped, Pose, Quaternion
from nav_msgs.msg import Path
from gazebo_msgs.srv import SpawnModel, SpawnModelRequest, SetModelState
from gazebo_msgs.msg import ModelState
import numpy as np

class LaneDemoNode(object):
    def __init__(self):
        rospy.init_node("lane_demo_node", anonymous=False)

        # 1) Define your waypoints (x, y) in world frame
        # self.waypoints = [
        #     (-6.5, 0.0), (-6.46, 0.47), (-6.35, 0.93), (-6.17, 1.36),
        #     (-5.93, 1.76), (-5.62, 2.12), (-5.26, 2.43), (-4.86, 2.67),
        #     (-4.43, 2.85), (-3.97, 2.96), (-3.5, 3.0), (-3.03, 2.96),
        #     (-2.57, 2.85), (-2.14, 2.67), (-1.74, 2.43), (-1.38, 2.12),
        #     (-1.07, 1.76), (-0.83, 1.36), (-0.65, 0.93), (-0.54, 0.47),
        #     (-0.5, 0.0), (0.54, -0.47), (0.65, -0.93), (0.83, -1.36),
        #     (1.07, -1.76), (1.38, -2.12), (1.74, -2.43), (2.14, -2.67),
        #     (2.57, -2.85), (3.03, -2.96), (3.5, -3.0), (3.97, -2.96),
        #     (4.43, -2.85), (4.86, -2.67), (5.26, -2.43), (5.62, -2.12),
        #     (5.93, -1.76), (6.17, -1.36), (6.35, -0.93), (6.46, -0.47),
        #     (6.5, 0.0)
        # ]

        # Circle params
        r = 5.0          # radius of each circle
        n = 40           # number of points per loop
        cx1, cy1 = -r, 0 # center of left circle
        cx2, cy2 = r, 0  # center of right circle

        # First loop (left circle, counterclockwise from bottom)
        theta1 = np.linspace(-np.pi/2, 3*np.pi/2, n, endpoint=False)
        left_loop = [(cx1 + r * np.cos(t), cy1 + r * np.sin(t)) for t in theta1]

        # Second loop (right circle, clockwise from top)
        theta2 = np.linspace(np.pi/2, -3*np.pi/2, n, endpoint=False)
        right_loop = [(cx2 + r * np.cos(t), cy2 + r * np.sin(t)) for t in theta2]

        # Combine for full figure 8
        self.waypoints = left_loop + right_loop


        # 2) How far from the center‐line the lane lines should be (meters)
        self.lane_half_width = 0.5  # so total lane width = 1.0 m

        # 3) Publisher for nav_msgs/Path
        self.path_pub = rospy.Publisher("/planned_path", Path, queue_size=1)

        # 4) Wait for Gazebo services
        rospy.loginfo("Waiting for /gazebo/spawn_sdf_model service...")
        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        self.spawn_srv = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)

        rospy.loginfo("Waiting for /gazebo/set_model_state service...")
        rospy.wait_for_service("/gazebo/set_model_state")
        self.set_state_srv = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        # 5) Spawn the two lane‐line boxes per segment
        self.spawn_lane_markings()

        # 6) Publish the Path (once)
        self.publish_path()

        # 7) Prepare to move the camera (or car) along that path
        self.planned_path = self.current_path.poses
        self.time_index = 0
        # Change this to your actual vehicle/model name
        self.camera_model_name = "camera_model"

        # 8) Timer: teleport model along path at 20 Hz
        self.timer = rospy.Timer(rospy.Duration(0.05), self.move_camera_along_path)

        rospy.loginfo("LaneDemoNode initialized.")

    def spawn_lane_markings(self):
        """
        For each consecutive waypoint segment, spawn exactly two
        thin white boxes (length = segment length) offset by lane_half_width.
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

            # 2) Find midpoint of this segment
            mid_x = (x0 + x1) / 2.0
            mid_y = (y0 + y1) / 2.0

            # 3) Compute normal vector (unit) to offset left/right
            nx = -math.sin(heading)
            ny = math.cos(heading)

            # 4) Compute left and right offset centers
            left_center_x = mid_x + nx * self.lane_half_width
            left_center_y = mid_y + ny * self.lane_half_width
            right_center_x = mid_x - nx * self.lane_half_width
            right_center_y = mid_y - ny * self.lane_half_width

            # 5) Box dimensions: 
            #    - x‐axis (width)  = 0.1 m (thin stripe)
            #    - y‐axis (length) = segment_length (so it exactly spans)
            #    - z‐axis (height) = 0.01 m (flush with ground)
            sx = segment_length
            sy = 0.1
            sz = 0.01

            # 6) Box color: solid white (RGBA)
            r, g, b, a = 1.0, 1.0, 1.0, 1.0

            box_sdf_template = """
            <?xml version="1.0" ?>
            <sdf version="1.6">
              <model name="{model_name}">
                <static>true</static>
                <link name="link">
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

            # 7) Spawn both “left” and “right” boxes
            for side, (cx, cy) in [
                ("left", (left_center_x, left_center_y)),
                ("right", (right_center_x, right_center_y))
            ]:
                model_name = "lane_line_{}_{}".format(side, i)
                half_z = sz / 2.0
                box_sdf = box_sdf_template.format(
                    model_name=model_name,
                    half_z=half_z,
                    sx=sx, sy=sy, sz=sz,
                    r=r, g=g, b=b, a=a
                )

                # Build the SpawnModelRequest
                req = SpawnModelRequest()
                req.model_name = model_name
                req.model_xml = box_sdf
                req.robot_namespace = ""
                req.reference_frame = "world"

                # Put the box center at (cx, cy, 0) and rotate by heading
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
                        rospy.loginfo("Spawned {}".format(model_name))
                    else:
                        rospy.logwarn("Failed to spawn {}: {}".format(model_name, res.status_message))
                except rospy.ServiceException as e:
                    rospy.logerr("Service call failed: {}".format(e))

    def publish_path(self):
        """
        Publish a nav_msgs/Path that goes exactly through each waypoint,
        with orientation set so that the vehicle faces “forward” along each segment.
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
        # For the last waypoint, reuse the final segment's heading:
        headings.append(headings[-1])

        for i, (x, y) in enumerate(self.waypoints):
            ps = PoseStamped()
            ps.header.stamp = rospy.Time.now()
            ps.header.frame_id = "map"
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.position.z = 0.0

            # Use the precomputed heading to build a quaternion:
            yaw = headings[i]
            q = tf.transformations.quaternion_from_euler(0, 0, yaw)
            ps.pose.orientation = Quaternion(q[0], q[1], q[2], q[3])

            path_msg.poses.append(ps)

        self.current_path = path_msg
        self.path_pub.publish(path_msg)
        rospy.loginfo("Published Path with {} waypoints (including orientation)".format(len(self.waypoints)))

    def move_camera_along_path(self, event):
        """
        Every timer tick, teleport “camera_model” (or car) to the next Pose in planned_path,
        including both position and orientation.
        """
        if not hasattr(self, "planned_path") or len(self.planned_path) == 0:
            return

        idx = self.time_index
        if idx >= len(self.planned_path):
            idx = len(self.planned_path) - 1

        target_pose = self.planned_path[idx].pose

        state_msg = ModelState()
        state_msg.model_name = self.camera_model_name
        state_msg.pose = Pose()
        state_msg.pose.position = target_pose.position
        # Copy over the orientation we stored in 'publish_path()'
        state_msg.pose.orientation = target_pose.orientation

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

        # Advance index and wrap around if needed
        self.time_index += 1
        if self.time_index >= len(self.planned_path):
            self.time_index = 0

def main():
    try:
        node = LaneDemoNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()
