#!/bin/bash
SESSION=sim

# Start new tmux session (detached)
# tmux new-session -d -s $SESSION -n Gazebo \
#   "roslaunch gazebo_ros empty_world.launch world_name:=$(rospack find lane_marking_demo)/worlds/random_trial.world use_sim_time:=false"

# Window 2: spawn random pose
tmux new-session -d -s $SESSION -n Spawn \
  "cd ~/catkin_ws && ./src/lane_marking_demo/src/spawn_random_pose.py"

# Window 3: mask publisher
tmux new-window -t $SESSION:2 -n MaskPublisher \
  "rosrun lane_marking_demo mask_publisher_node.py"

# Window 4: export model
tmux new-window -t $SESSION:3 -n ExportModel \
  "python3 ~/catkin_ws/src/lane_marking_demo/dvs_learning/seg2cmd/training/export_model.py"

# Window 5: NN controller
tmux new-window -t $SESSION:4 -n NNController \
  "rosrun lane_marking_demo nn_controller_node.py"

# Window 6: lane demo spawn
tmux new-window -t $SESSION:5 -n LaneDemo \
  "rosrun lane_marking_demo lane_demo_spawn_node_circle.py"

# Window 7: camera info pub
tmux new-window -t $SESSION:6 -n CameraInfo \
  "rostopic pub --once --latch /lane_demo/front_cam/camera_info sensor_msgs/CameraInfo '{header: {frame_id: \"front_cam\"}, height: 128, width: 128, distortion_model: \"plumb_bob\", D: [0,0,0,0,0], K: [128.0,0.0,64.0,0.0,128.0,64.0,0.0,0.0,1.0], R: [1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0], P: [128.0,0.0,64.0,0.0,0.0,128.0,64.0,0.0,0.0,0.0,1.0,0.0]}'"

# Window 8: dvs renderer
tmux new-window -t $SESSION:7 -n DVSRenderer \
  "rosrun dvs_renderer dvs_renderer events:=/lane_demo/front_cam/events camera_info:=/lane_demo/front_cam/camera_info"

# Window 9: throttle dvs
tmux new-window -t $SESSION:8 -n Throttle \
  "rosrun topic_tools throttle messages /dvs_rendering 30.0"

# Window 10: ackermann to cmdvel
tmux new-window -t $SESSION:9 -n Ackermann2CmdVel \
  "rosrun lane_marking_demo ackermann_to_cmdvel.py"

# Window 11: rviz
tmux new-window -t $SESSION:10 -n RViz \
  "rviz"

# Attach to the session
tmux attach -t $SESSION
