import random
from pathlib import Path
import numpy as np
import argparse

WORLD_TEMPLATE_HEADER = """<?xml version='1.7'?>
<sdf version='1.7'>
  <world name='default'>
    <gravity>0 0 -9.8</gravity>
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>
"""

WORLD_TEMPLATE_FOOTER = """
  </world>
</sdf>
"""

SHAPES = ["box", "sphere", "cylinder"]

def generate_model(name, shape, pose):
    if shape == "box":
        geometry = "<box><size>1 1 1</size></box>"
    elif shape == "sphere":
        geometry = "<sphere><radius>0.5</radius></sphere>"
    elif shape == "cylinder":
        geometry = "<cylinder><radius>0.5</radius><length>1</length></cylinder>"
    else:
        raise ValueError(f"Unsupported shape: {shape}")

    return f"""
    <model name='{name}'>
      <pose>{pose}</pose>
      <static>0</static>
      <link name='link'>
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
    """

def generate_random_pose(x_range, y_range, z=0.5, min_dist_to_track=1.5, waypoints=None, max_tries=100):
    for _ in range(max_tries):
        x = round(random.uniform(*x_range), 3)
        y = round(random.uniform(*y_range), 3)
        if waypoints is not None:
            distances = np.linalg.norm(waypoints - np.array([x, y]), axis=1)
            if np.min(distances) < min_dist_to_track:
                continue  # Too close to the track
        return f"{x} {y} {z} 0 0 0"
    return None  # Failed to find a good pose

def generate_world(output_path, num_objects_range=(20, 40), x_range=(-20, 20), y_range=(-20, 20), waypoints=None):
    models = []
    num_objects = random.randint(*num_objects_range)

    # Load centerline
    waypoint_path = Path.home() / "catkin_ws" / "src" / "lane_marking_demo" / "worlds" / f"track_{track_name}.npy"
    waypoints = np.load(waypoint_path)
    
    for i in range(num_objects):
        shape = random.choice(SHAPES)
        pose = generate_random_pose(x_range, y_range, z=0.5, min_dist_to_track=1.5, waypoints=waypoints)
        if pose is None:
            continue  # Could not place this object safely
        model_str = generate_model(f"{shape}_{i}", shape, pose)
        models.append(model_str)

    full_world = WORLD_TEMPLATE_HEADER + "".join(models) + WORLD_TEMPLATE_FOOTER

    with open(output_path, "w") as f:
        f.write(full_world)

    print(f"Random world generated with {len(models)} objects: {output_path}")

# Example usage
# 1. Parse CLI argument
parser = argparse.ArgumentParser()
parser.add_argument("--track", type=str, default="circle", choices=["circle", "figure8", "square_tight", "square_smooth", "square_random"])
args = parser.parse_args()
track_name = args.track

if __name__ == "__main__":
    # 2. Load the correct track
    output_world = output_world = Path.home() / "catkin_ws" / "src" / "lane_marking_demo" / "worlds" / "random_trial.world"
    generate_world(output_world)

    print(f"World generated with {track_name} track.")

    # Optional: Launch Gazebo with this world
    # import os
    # os.system(f"roslaunch gazebo_ros empty_world.launch world_name:={output_world}")
