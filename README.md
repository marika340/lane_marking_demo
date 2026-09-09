# vijaylab_car_summer2025

## Note
This is inspired from [evfly](https://openreview.net/pdf?id=82bpTugrMt) and [vitfly](https://arxiv.org/abs/2405.10391) works, and some concepts/codes were borrowed from them. 

## How to use 
Run the following commands. 

### Make .world files
1. `roslaunch gazebo_ros empty_world.launch`

2. `rosrun lane_marking_demo lane_demo_spawn_node_${TRACK}.py`

3. After the lane spawning is completed, save the world as a .world file in gazebo GUI. It takes time to save the gazebo world, so make sure it's saved before closing the gazebo window. 

4. Fix the camera pose. This determines the perspective. 

### Data Collection
1. `cd ~/catkin_ws/src/lane_marking_demo/bin`

2. Collect datasets (automated)

Use config

`./collect_datasets_from_config.sh`

Whole course

`./collect_datasets_whole_course.sh`

Starting part
`./collect_datasets_starting_part.sh`

### Velocity control
1. `git clone` this repository under `catkin_ws/src`

2. Prepare basic gazebo environment

If you want random scenes, this is how you create them. 

`python3 ~/catkin_ws/src/lane_marking_demo/src/generate_random_world.py --track circle`

`python3 ~/catkin_ws/src/lane_marking_demo/src/generate_random_world.py --track figure8`

`python3 ~/catkin_ws/src/lane_marking_demo/src/generate_random_world.py --track square_tight`

`python3 ~/catkin_ws/src/lane_marking_demo/src/generate_random_world.py --track square_smooth`

We have to set `use_sim_time:=false` so that dataset collects rostime as timestamp later. 

`roslaunch gazebo_ros empty_world.launch world_name:=$(rospack find lane_marking_demo)/worlds/random_trial.world use_sim_time:=false`

If you want to use predetermined scenes, we have multiple scenes (scene 1: half crowded half empty, scene 2: crowded everywhere, scene 3: sparse). Note: Mask needs to be adjusted on desktop because of the sky color. The mask includes the border between the ground and the sky in the current settings. 

`roslaunch gazebo_ros empty_world.launch world_name:=$(rospack find lane_marking_demo)/worlds/scene1.world use_sim_time:=false`

`roslaunch gazebo_ros empty_world.launch world_name:=$(rospack find lane_marking_demo)/worlds/scene2.world use_sim_time:=false`

`roslaunch gazebo_ros empty_world.launch world_name:=$(rospack find lane_marking_demo)/worlds/scene3.world use_sim_time:=false`

If you want to randomize initial position and orientation, run this command. 

`./src/lane_marking_demo/src/spawn_random_pose.py`

Instead, if you want to manually set initial position and orientation, change `-x 0 -y 0 -z 0 -R 0 -P 0 -Y 0` to desirable values.

`rosrun gazebo_ros spawn_model -file ~/catkin_ws/src/lane_marking_demo/worlds/camera_model_synchronized.sdf -sdf -model camera_model -x 0 -y 0 -z 0 -R 0 -P 0 -Y 0`

3. Draw lanes and let a car drive between lanes

lane_demo_node_vel_control_{track_shape}.py attempts to drive the car between the two lanes using velocity control.

`rosrun lane_marking_demo lane_demo_node_vel_control_circle.py`

`rosrun lane_marking_demo lane_demo_node_vel_control_figure8.py`

`rosrun lane_marking_demo lane_demo_node_vel_control_square_tight.py`

`rosrun lane_marking_demo lane_demo_node_vel_control_square_smooth.py`

4. Visualize dvs camera data

`rostopic pub --once --latch /lane_demo/front_cam/camera_info sensor_msgs/CameraInfo "{header: {frame_id: 'front_cam'}, height: 128, width: 128, distortion_model: 'plumb_bob', D: [0, 0, 0, 0, 0], K: [128.0, 0.0, 64.0, 0.0, 128.0, 64.0, 0.0, 0.0, 1.0], R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], P: [128.0, 0.0, 64.0, 0.0, 0.0, 128.0, 64.0, 0.0, 0.0, 0.0, 1.0, 0.0]}"`

`rosrun dvs_renderer dvs_renderer  events:=/lane_demo/front_cam/events camera_info:=/lane_demo/front_cam/camera_info`

`rosrun topic_tools throttle messages /dvs_rendering 30.0`

`rviz -d ~/rviz/rgb_dvs_cam.rviz`

Add Image to the rviz panel. Select Topic /dvs_rendering. 

5. Image segmentation dataset collector

`rosrun lane_marking_demo lane_segmentation_dataset_node_2.py`

### Position control

1. `git clone` this repository under `catkin_ws/src`

2. Prepare basic gazebo environment

```roslaunch gazebo_ros empty_world.launch world_name:=$(rospack find lane_marking_demo)/worlds/lane_demo_random_objects_slow.world use_sim_time:=true```

3. Draw lanes and let a car drive between lanes

lane_demo_node.py draws 8-figure lanes and lets the car drive between the two lanes. The car drives on certain waypoints, which are in the middle of the two lanes. 

`rosrun lane_marking_demo lane_demo_node.py`

4. Visualize dvs camera data

`rostopic pub --once --latch /lane_demo/front_cam/camera_info sensor_msgs/CameraInfo "{header: {frame_id: 'front_cam'}, height: 128, width: 128, distortion_model: 'plumb_bob', D: [0, 0, 0, 0, 0], K: [128.0, 0.0, 64.0, 0.0, 128.0, 64.0, 0.0, 0.0, 1.0], R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], P: [128.0, 0.0, 64.0, 0.0, 0.0, 128.0, 64.0, 0.0, 0.0, 0.0, 1.0, 0.0]}"`

`rosrun dvs_renderer dvs_renderer  events:=/lane_demo/front_cam/events camera_info:=/lane_demo/front_cam/camera_info`

`rviz -d ~/rviz/rgb_dvs_cam.rviz`

Add Image to the rviz panel. Select Topic /dvs_rendering. 

5. Image segmentation dataset collector

`rosrun lane_marking_demo lane_segmentation_dataset_node_4.py`

### Training

#### event2cmd

`cd ~/catkin_ws/src/lane_marking_demo/dvs_learning/event2cmd/training`

`python3 train.py --config configs/train_config_event2cmd.txt`

`tensorboard --logdir ~/catkin_ws/src/lane_marking_demo/dvs_learning/event2cmd/training/log/`

#### event2seg (edited vitfly)

`cd ~/catkin_ws/src/lane_marking_demo/dvs_learning/event2seg/training`

`python3 train.py --config configs/train_config_event2seg.txt `

`tensorboard --logdir ~/catkin_ws/src/lane_marking_demo/dvs_learning/event2seg/training/log/`


#### seg2cmd (for this project)

`cd ~/catkin_ws/src/lane_marking_demo/dvs_learning/seg2cmd/training`

`python3 train.py --config configs/train_config_seg2cmd.txt`

`tensorboard --logdir ~/catkin_ws/src/lane_marking_demo/dvs_learning/seg2cmd/training/log/`

#### seg2cmd (vitfly)

`cd ~/catkin_ws/src/lane_marking_demo/dvs_learning/seg2cmd/training`

`python3 train.py --config configs/train_config_seg2cmd.txt`

<!-- `python3 train.py   --basedir /home/marikan/catkin_ws/src/lane_marking_demo   --dataset_range 110 171   --logdir logs   --model_type LSTMNetVIT   --epochs 100   --lr 1e-3   --device cuda` -->

`tensorboard --logdir ~/catkin_ws/src/lane_marking_demo/logs`

If you're using ssh, use the command below in the new terminal and see the tensorboard. 

`ssh -L 6006:localhost:6006 user@remote_host`

<!-- Publish in Ackermannmsgs

`roscore`

`rosrun lane_marking_demo seg_to_ackermann.py   _model_path:=/home/marikan/catkin_ws/src/lane_marking_demo/logs/d07_30_t16_37/model_050.pth   _mask_topic:=/lane_segmentation/mask   _cmd_topic:=/ackermann_cmd   _frame_id:=base_link` -->

### Evaluation
#### event2seg

`cd ~/catkin_ws/src/lane_marking_demo/dvs_learning/event2seg/training`

`python3 visualize.py --config configs/eval_config_event2seg.txt`

#### seg2cmd

`cd ~/catkin_ws/src/lane_marking_demo/dvs_learning/seg2cmd/training`

`python3 evaluation_tools.py --config configs/eval_config_seg2cmd.txt`

### NN-in-the-control (event2cmd)
`cd catkin_ws/src/lane_marking_demo/bin`

`./run_nn_in_the_control_loop_event2cmd.sh`

### NN-in-the-control (event2seg)
`cd catkin_ws/src/lane_marking_demo/bin`

`./run_nn_in_the_control_loop2.sh`

Alternatively, 

Terminal 1

`roslaunch gazebo_ros empty_world.launch world_name:=$(rospack find lane_marking_demo)/worlds/random_trial.world use_sim_time:=false`

Terminal 2

`cd catkin_ws`

`./src/lane_marking_demo/src/spawn_random_pose.py`

Terminal 3

`rosrun lane_marking_demo mask_publisher_node.py`

Terminal 4

`python3 catkin_ws/src/lane_marking_demo/dvs_learning/seg2cmd/training/export_model.py`

Terminal 5

`rosrun lane_marking_demo lane_demo_spawn_node_{track shape}.py`

Terminal 6

`rostopic pub --once --latch /lane_demo/front_cam/camera_info sensor_msgs/CameraInfo "{header: {frame_id: 'front_cam'}, height: 128, width: 128, distortion_model: 'plumb_bob', D: [0, 0, 0, 0, 0], K: [128.0, 0.0, 64.0, 0.0, 128.0, 64.0, 0.0, 0.0, 1.0], R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], P: [128.0, 0.0, 64.0, 0.0, 0.0, 128.0, 64.0, 0.0, 0.0, 0.0, 1.0, 0.0]}"`

Terminal 7

`rosrun dvs_renderer dvs_renderer  events:=/lane_demo/front_cam/events camera_info:=/lane_demo/front_cam/camera_info`

Terminal 8

`rosrun topic_tools throttle messages /dvs_rendering 30.0`

Terminal 9

`rosrun lane_marking_demo ackermann_to_cmdvel.py`

Terminal 10

`rosrun lane_marking_demo nn_controller_node.py`

`python3 nn_controller_node.py offline test1.png test2.png`

Terminal 11

`rviz`