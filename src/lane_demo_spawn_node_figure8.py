#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
import tf
import numpy as np

from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Path
from gazebo_msgs.srv import SpawnModel, SpawnModelRequest, GetModelState
from std_msgs.msg import Bool


class LaneDemoSpawnNodeCircle(object):
    def __init__(self):
        rospy.init_node("lane_demo_spawn_node_figure8", anonymous=False)

        # 1) Define waypoints (circle track)
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

        # Publishers
        self.path_pub = rospy.Publisher("/planned_path", Path, queue_size=1, latch=True)
        self.ready_pub = rospy.Publisher("/lane_demo/ready", Bool, queue_size=1, latch=True)

        # Gazebo services
        rospy.loginfo("Waiting for /gazebo/spawn_sdf_model...")
        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        self.spawn_srv = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)

        rospy.loginfo("Waiting for /gazebo/get_model_state...")
        rospy.wait_for_service("/gazebo/get_model_state")
        self.get_state_srv = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)

        # Spawn lane markings
        self.spawn_lane_markings()

        # Publish path (once)
        self.publish_path()

        # Publish ready flag
        rospy.sleep(1.0)
        self.ready_pub.publish(Bool(data=True))
        rospy.loginfo("LaneDemoSpawnNodeCircle initialized (spawned lanes, published path, set ready=True).")

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


def main():
    try:
        LaneDemoSpawnNodeCircle()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
