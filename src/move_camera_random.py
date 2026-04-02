#!/usr/bin/env python3
import random
import math
import os
import numpy as np
import tf
import rospy
import argparse
from gazebo_msgs.srv import DeleteModel, SpawnModel, SetModelState
from geometry_msgs.msg import Pose, Twist
from gazebo_msgs.msg import ModelStates, ModelState

MODEL_NAME = "camera_model"
MODEL_PATH = "/home/marikan/catkin_ws/src/lane_marking_demo/worlds/camera_model_synchronized.sdf"

def model_exists(name):
    """Check if model already exists in Gazebo"""
    states = rospy.wait_for_message("/gazebo/model_states", ModelStates)
    return name in states.name


# def safe_delete_model(name):
#     try:
#         rospy.wait_for_service("/gazebo/delete_model", timeout=5)
#         delete_model = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
#         resp = delete_model(name)
#         if resp.success:
#             rospy.loginfo(f"Deleted {name}")
#         else:
#             rospy.logwarn(f"Delete {name} failed: {resp.status_message}")
#     except rospy.ServiceException as e:
#         rospy.logwarn(f"Delete service call failed: {e}")
#     except rospy.ROSException:
#         rospy.logwarn("Delete service not available (Gazebo restarting?)")

# def delete_existing_model():
#     if model_exists(MODEL_NAME):
#         safe_delete_model(MODEL_NAME)
#         rospy.sleep(1.0)  # let Gazebo settle
#     else:
#         rospy.loginfo("camera_model not found, skipping delete")


# def delete_existing_model():
#     rospy.wait_for_service("/gazebo/delete_model")
#     delete_model = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
#     # try:
#     #     delete_model(MODEL_NAME)
#     #     rospy.loginfo(f"Deleted existing {MODEL_NAME}")
#     # except rospy.ServiceException:
#     #     rospy.logwarn("camera_model not found to delete, skipping")
#     if model_exists(MODEL_NAME):
#         delete_model(MODEL_NAME)
#         rospy.loginfo("Deleted camera_model")
#     else:
#         rospy.logwarn("camera_model not found, skipping delete")


def compute_random_pose(track, waypoints, lane_half_width=0.2):
    idx = random.randint(0, len(waypoints) - 2)
    x0, y0 = waypoints[idx]
    x1, y1 = waypoints[idx + 1]

    # Heading along the path
    dx, dy = x1 - x0, y1 - y0
    route_yaw = math.atan2(dy, dx)

    # Normal vector
    nx, ny = -math.sin(route_yaw), math.cos(route_yaw)

    # Lateral offset
    # offset = random.choice([-1, 1]) * random.uniform(0.0, 2.0 * lane_half_width)
    offset = random.choice([-1,1]) * random.uniform(0.0, range * lane_half_width)
    x = x0 + nx * offset
    y = y0 + ny * offset
    z = height
    roll = pitch = 0.0

    # Track-specific yaw logic
    if track == "figure8":
        # if (x0 < 0):
        #     if offset > lane_half_width:
        #         yaw = route_yaw - random.uniform(0.0, math.pi / 3.0)
        #     elif offset > 0:
        #         yaw = route_yaw - random.uniform(0.0, math.pi / 6.0)
        #     elif offset > -lane_half_width:
        #         yaw = route_yaw + random.uniform(0.0, math.pi / 6.0)
        #     else:
        #         yaw = route_yaw + random.uniform(0.0, math.pi / 3.0)
        # else:
        #     if offset > lane_half_width:
        #         yaw = route_yaw + random.uniform(0.0, math.pi / 3.0)
        #     elif offset > 0:
        #         yaw = route_yaw + random.uniform(0.0, math.pi / 6.0)
        #     elif offset > -lane_half_width:
        #         yaw = route_yaw - random.uniform(0.0, math.pi / 6.0)
        #     else:
        #         yaw = route_yaw - random.uniform(0.0, math.pi / 3.0)
        if offset > lane_half_width:
            yaw = route_yaw - random.uniform(0.0, math.pi / 3.0)
        elif offset > 0:
            yaw = route_yaw - random.uniform(0.0, math.pi / 6.0)
        elif offset > -lane_half_width:
            yaw = route_yaw + random.uniform(0.0, math.pi / 6.0)
        else:
            yaw = route_yaw + random.uniform(0.0, math.pi / 3.0)

    elif track in ["circle", "square_smooth", "square_random", "figure8_opposite"]:
        if offset > lane_half_width:
            yaw = route_yaw - random.uniform(0.0, math.pi / 3.0)
        elif offset > 0:
            yaw = route_yaw - random.uniform(0.0, math.pi / 6.0)
        elif offset > -lane_half_width:
            yaw = route_yaw + random.uniform(0.0, math.pi / 6.0)
        else:
            yaw = route_yaw + random.uniform(0.0, math.pi / 3.0)
    else:
        yaw = route_yaw  # fallback

    rospy.loginfo(f"[{track}] Spawn pose: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")
    return x, y, z, roll, pitch, yaw

# def spawn_model_at_pose(x, y, z, roll, pitch, yaw):
#     rospy.wait_for_service("/gazebo/spawn_sdf_model")
#     spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)

#     with open(MODEL_PATH, "r") as f:
#         model_xml = f.read()

#     q = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
#     pose = Pose()
#     pose.position.x, pose.position.y, pose.position.z = x, y, z
#     pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q

#     try:
#         spawn_model(MODEL_NAME, model_xml, "", pose, "world")
#         rospy.loginfo(f"Spawned {MODEL_NAME} at ({x:.2f}, {y:.2f}, yaw={yaw:.2f})")
#     except Exception as e:
#         rospy.logerr(f"Failed to spawn {MODEL_NAME}: {e}")

def spawn_if_needed():
    """Spawn the camera model only once if it doesn't exist yet."""
    if model_exists(MODEL_NAME):
        rospy.loginfo(f"{MODEL_NAME} already exists, not respawning.")
        return

    rospy.wait_for_service("/gazebo/spawn_sdf_model")
    spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)

    with open(MODEL_PATH, "r") as f:
        model_xml = f.read()

    pose = Pose()
    pose.position.x = 0.0
    pose.position.y = 0.0
    pose.position.z = 0.0
    pose.orientation.w = 1.0

    try:
        spawn_model(MODEL_NAME, model_xml, "", pose, "world")
        rospy.loginfo(f"Spawned {MODEL_NAME} at origin")
    except Exception as e:
        rospy.logerr(f"Failed to spawn {MODEL_NAME}: {e}")

def set_model_pose(x, y, z, roll, pitch, yaw):
    """Teleport existing model to new pose using /set_model_state."""
    rospy.wait_for_service("/gazebo/set_model_state")
    set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

    q = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = x, y, z
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q

    state = ModelState()
    state.model_name = MODEL_NAME
    state.pose = pose

    # Reset a vehicle so it doesn't drift
    state.twist.linear.x = 0.0
    state.twist.linear.y = 0.0
    state.twist.linear.z = 0.0
    state.twist.angular.x = 0.0
    state.twist.angular.y = 0.0
    state.twist.angular.z = 0.0
    state.reference_frame = ""
    
    try:
        set_state(state)
        rospy.loginfo(f"Moved {MODEL_NAME} to ({x:.2f}, {y:.2f}, yaw={yaw:.2f})")
    except Exception as e:
        rospy.logerr(f"Failed to set state of {MODEL_NAME}: {e}")


if __name__ == "__main__":
    rospy.init_node("move_camera_random")

    parser = argparse.ArgumentParser()
    parser.add_argument("--track", type=str, default="figure8",
                        choices=["circle", "figure8", "figure8_opposite", "square_smooth", "square_random"])
    parser.add_argument("--range", type=float, default=1.0)
    parser.add_argument("--height", type=float, default=0.0)
    args = parser.parse_args()

    track = args.track
    range = args.range
    height = args.height
    waypoints = np.load(f"/home/marikan/catkin_ws/src/lane_marking_demo/worlds/track_{track}.npy")

    # Ensure model exists once
    spawn_if_needed()

    # delete_existing_model()
    # safe_delete_model(MODEL_NAME)
    rospy.sleep(0.05)  # 50 ms pause
    x, y, z, roll, pitch, yaw = compute_random_pose(track, waypoints)
    set_model_pose(x, y, z, roll, pitch, yaw)

    stop_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1, latch=True)
    rospy.sleep(0.1)  # give publisher time to connect
    stop_pub.publish(Twist())  # all zeros
    # spawn_model_at_pose(x, y, z, roll, pitch, yaw)
