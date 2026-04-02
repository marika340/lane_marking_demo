#!/usr/bin/env python3
import rospy
import random
import numpy as np
from pathlib import Path
from gazebo_msgs.srv import DeleteModel, SpawnModel
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose
import tf
import time

SHAPES = ["box", "sphere", "cylinder"]

def generate_random_pose(x_range, y_range, z=0.5, min_dist_to_track=1.5, waypoints=None, max_tries=100):
    for _ in range(max_tries):
        x = round(random.uniform(*x_range), 3)
        y = round(random.uniform(*y_range), 3)
        if waypoints is not None:
            distances = np.linalg.norm(waypoints - np.array([x, y]), axis=1)
            if np.min(distances) < min_dist_to_track:
                continue  # Too close to the track
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        q = tf.transformations.quaternion_from_euler(0, 0, 0)
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
        return pose
    return None

def generate_shape_sdf(shape, name):
    if shape == "box":
        geometry = "<box><size>1 1 1</size></box>"
    elif shape == "sphere":
        geometry = "<sphere><radius>0.5</radius></sphere>"
    elif shape == "cylinder":
        geometry = "<cylinder><radius>0.5</radius><length>1</length></cylinder>"
    else:
        raise ValueError(f"Unsupported shape: {shape}")

    return f"""
    <?xml version="1.0" ?>
    <sdf version="1.6">
      <model name='{name}'>
        <static>false</static>
        <link name='link'>
          <pose>0 0 0 0 0 0</pose>
          <inertial><mass>1</mass></inertial>
          <collision name='collision'>
            <geometry>{geometry}</geometry>
          </collision>
          <visual name='visual'>
              <geometry>{geometry}</geometry>
              <material>
                  <script>
                  <name>Gazebo/Grey</name>
                  <uri>file://media/materials/scripts/gazebo.material</uri>
                  </script>
              </material>
          </visual>
        </link>
      </model>
    </sdf>
    """

def delete_old_obstacles(max_count=100):
    rospy.wait_for_service("/gazebo/delete_model")
    delete_model = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)

    # Get list of current models
    model_states = rospy.wait_for_message("/gazebo/model_states", ModelStates)
    models_to_delete = [name for name in model_states.name if any(s in name for s in SHAPES)]

    for name in models_to_delete:
        try:
            delete_model(name)
            rospy.loginfo(f"Deleted {name}")
            time.sleep(0.05)
        except rospy.ServiceException:
            pass

def spawn_obstacles(track_name="figure8", num_objects_range=(20, 40), x_range=(-20, 20), y_range=(-20, 20)):
    rospy.wait_for_service("/gazebo/spawn_sdf_model")
    spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
    # if not rospy.wait_for_service("/gazebo/spawn_sdf_model", timeout=10):
    #     rospy.logerr("Gazebo not available, aborting obstacle reset")
    #     return
    try:
        rospy.wait_for_service("/gazebo/spawn_sdf_model", timeout=10)
        spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
    except rospy.ROSException:
        rospy.logerr("Gazebo not available, aborting obstacle reset")
        return


    # Load waypoints for distance constraint
    waypoint_path = Path.home() / "catkin_ws" / "src" / "lane_marking_demo" / "worlds" / f"track_{track_name}.npy"
    waypoints = np.load(waypoint_path)

    num_objects = random.randint(*num_objects_range)
    count = 0

    for i in range(num_objects):
        shape = random.choice(SHAPES)
        pose = generate_random_pose(x_range, y_range, z=0.5, min_dist_to_track=1.5, waypoints=waypoints)
        if pose is None:
            continue
        name = f"{shape}_{i}"
        sdf = generate_shape_sdf(shape, name)
        try:
            spawn_model(name, sdf, "", pose, "world")
            rospy.loginfo(f"Spawned {name} at ({pose.position.x:.2f}, {pose.position.y:.2f})")
            count += 1
        except Exception as e:
            rospy.logerr(f"Failed to spawn {name}: {e}")
    rospy.loginfo(f"Spawned {count} obstacles.")

if __name__ == "__main__":
    import argparse
    rospy.init_node("reset_obstacles")

    parser = argparse.ArgumentParser()
    parser.add_argument("--track", type=str, default="figure8",
                        choices=["circle", "figure8", "square_tight", "square_smooth", "square_random"])
    args = parser.parse_args()

    delete_old_obstacles()
    spawn_obstacles(track_name=args.track)
