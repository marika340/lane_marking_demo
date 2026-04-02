#!/usr/bin/env python3
import random
import math
import os
import numpy as np
import tf

# Load waypoints for the desired track
# figure 8
waypoints = np.load("/home/marikan/catkin_ws/src/lane_marking_demo/worlds/track_figure8.npy")

# Pick up a random waypoint
idx = random.randint(0, len(waypoints)-2)
x0, y0 = waypoints[idx]
x1, y1 = waypoints[idx+1]

# Compute a heading along the path
dx = x1 - x0
dy = y1 - y0
route_yaw = math.atan2(dy, dx)

# Normal to the path
nx, ny = -math.sin(route_yaw), math.cos(route_yaw)

lane_half_width = 0.2
offset = random.choice([-1, 1]) * random.uniform(0.0, 2.0 * lane_half_width)
x = x0 + nx * offset
y = y0 + ny * offset
print(nx*offset)
print(ny*offset)
z = 0

roll = 0
pitch = 0

# figure 8
if (x0 < 0):
    if offset > lane_half_width:
        yaw = route_yaw - random.uniform(0.0, math.pi/3.0)
    elif (offset <= lane_half_width) and (offset > 0):
        yaw = route_yaw - random.uniform(0.0, math.pi/6.0)
    elif (offset > -lane_half_width) and (offset <= 0):
        yaw = route_yaw + random.uniform(0.0, math.pi/6.0)
    else:  # offset <= -lane_half_width
        yaw = route_yaw + random.uniform(0.0, math.pi/3.0)
else:
    if offset > lane_half_width:
        yaw = route_yaw + random.uniform(0.0, math.pi/3.0)
    elif (offset <= lane_half_width) and (offset > 0):
        yaw = route_yaw + random.uniform(0.0, math.pi/6.0)
    elif (offset > -lane_half_width) and (offset <= 0):
        yaw = route_yaw - random.uniform(0.0, math.pi/6.0)
    else:  # offset <= -lane_half_width
        yaw = route_yaw - random.uniform(0.0, math.pi/3.0)

# square, figure 8 opposite
# if offset > lane_half_width:
#     yaw = route_yaw - random.uniform(0.0, math.pi/3.0)
# elif (offset <= lane_half_width) and (offset > 0):
#     yaw = route_yaw - random.uniform(0.0, math.pi/6.0)
# elif (offset > -lane_half_width) and (offset <= 0):
#     yaw = route_yaw + random.uniform(0.0, math.pi/6.0)
# else:  # offset <= -lane_half_width
#     yaw = route_yaw + random.uniform(0.0, math.pi/3.0)

# Generate random x in (9, 11)
# circle
# center_x = 10
# figure 8
# center_x = 0
# figure 8 opposite
# x = 0
# square tight
# x = -4.5
# square smooth
# x = -3.0
# x = random.uniform(center_x - 1, center_x + 1)

# y and z are fixed (you can randomize y if needed)
# circle, figure 8
# y = 0
# figure 8 opposite
# center_y = 0
# square tight, square smooth
# center_y = -5.0
# y = random.uniform(center_y - 1, center_y + 1)
# z = 0

# Fixed roll and pitch
# roll = 0
# pitch = 0

# Random yaw in (-π/3, π/3)
# radians
# circle
# route_yaw = math.pi/2
# figure 8
# route_yaw = 0.785
# figure 8 opposite direction
# route_yaw = 2.356  # 135 deg = pi*3/4
# square tight, square smooth
# route_yaw = 0

# circle, figure8
# if x<center_x:
#   if x>center_x-0.5:
#     yaw = random.uniform(route_yaw - math.pi/6, route_yaw)
#   else:
#     yaw = random.uniform(route_yaw - math.pi/3, route_yaw)
# else:
#   if x<center_x+0.5:
#     yaw = random.uniform(route_yaw, route_yaw + math.pi/6)
#   else:
#     yaw = random.uniform(route_yaw, route_yaw + math.pi/3)

# figure8 opposite
# if y<center_y:
#   if y>center_y-0.5:
#     yaw = random.uniform(route_yaw - math.pi/6, route_yaw)
#   else:
#     yaw = random.uniform(route_yaw - math.pi/3, route_yaw)
    
# else:
#   if y<center_y+0.5:
#     yaw = random.uniform(route_yaw, route_yaw + math.pi/6)
#   else:
#     yaw = random.uniform(route_yaw, route_yaw + math.pi/3)

# square tight, square smooth
# if y<center_y:
#   if y>center_y-0.5:
#     yaw = random.uniform(route_yaw, route_yaw + math.pi/6)
#   else:
#     yaw = random.uniform(route_yaw, route_yaw + math.pi/3)
    
# else:
#   if y<center_y+0.5:
#     yaw = random.uniform(route_yaw - math.pi/6, route_yaw)
#   else:
#     yaw = random.uniform(route_yaw - math.pi/3, route_yaw)


# Format the spawn_model command
cmd = f"""
rosrun gazebo_ros spawn_model \\
  -file ~/catkin_ws/src/lane_marking_demo/worlds/camera_model_synchronized.sdf \\
  -sdf -model camera_model \\
  -x {x:.3f} -y {y:.3f} -z {z:.3f} \\
  -R {roll:.3f} -P {pitch:.3f} -Y {yaw:.3f}
"""

# Print and execute the command
print("Running command:")
print(cmd)
os.system(cmd)
