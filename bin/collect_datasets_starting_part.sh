#!/bin/bash

NUM_RUNS=10

for ((i=0; i<$NUM_RUNS; i++)); do
    echo "========================"
    echo "Starting trial $i"
    echo "========================"

    # Step 1: Generate random world
    # gnome-terminal -- bash -c "
    #     python3 ~/catkin_ws/src/lane_marking_demo/src/generate_random_world.py --track circle;
    #     exec bash
    # "
    # gnome-terminal -- bash -c "
    #     python3 ~/catkin_ws/src/lane_marking_demo/src/generate_random_world.py --track figure8;
    #     exec bash
    # "
    gnome-terminal -- bash -c "
        python3 ~/catkin_ws/src/lane_marking_demo/src/generate_random_world.py --track square_smooth;
        exec bash
    "
    sleep 2

    # Step 2: Launch Gazebo
    gnome-terminal -- bash -c "
        roslaunch gazebo_ros empty_world.launch world_name:=$(rospack find lane_marking_demo)/worlds/random_trial.world use_sim_time:=true;
        exec bash
    "
    sleep 5

    # Step 3: Spawn random vehicle pose
    gnome-terminal -- bash -c "
        cd ~/catkin_ws && python3 ./src/lane_marking_demo/src/spawn_random_pose.py;
        exec bash
    "
    sleep 2

    # Step 4: Start lane drawing + control node
    # gnome-terminal -- bash -c "
    #     rosrun lane_marking_demo lane_demo_node_vel_control_circle.py;
    #     exec bash
    # "
    # gnome-terminal -- bash -c "
    #     rosrun lane_marking_demo lane_demo_node_vel_control_figure8.py;
    #     exec bash
    # "
    gnome-terminal -- bash -c "
        rosrun lane_marking_demo lane_demo_node_vel_control_square_smooth.py;
        exec bash
    "
    sleep 2

    # Step 5: Publish camera info
    gnome-terminal -- bash -c "
        rostopic pub --once --latch /lane_demo/front_cam/camera_info sensor_msgs/CameraInfo \"{header: {frame_id: 'front_cam'}, height: 128, width: 128, distortion_model: 'plumb_bob', D: [0, 0, 0, 0, 0], K: [128.0, 0.0, 64.0, 0.0, 128.0, 64.0, 0.0, 0.0, 1.0], R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], P: [128.0, 0.0, 64.0, 0.0, 0.0, 128.0, 64.0, 0.0, 0.0, 0.0, 1.0, 0.0]}\";
        exec bash
    "
    sleep 2

    # Step 6: Start DVS renderer
    gnome-terminal -- bash -c "
        rosrun dvs_renderer dvs_renderer events:=/lane_demo/front_cam/events camera_info:=/lane_demo/front_cam/camera_info;
        exec bash
    "
    sleep 2

    # Step 7: Start dataset collector (blocking)
    gnome-terminal -- bash -c "
        rosrun lane_marking_demo lane_segmentation_dataset_node_6.py;
        exec bash
    "

    while ! rosnode list | grep -q "/lane_segmentation_dataset_node_6"; do
        sleep 1
    done

    echo "[Trial $i] Waiting for /lane_demo/ready signal..."
    python3 ~/catkin_ws/src/lane_marking_demo/src/wait_for_lane_ready.py
    # echo "[Trial $i] Waiting for /lane_segmentation_dataset_node_6 to appear..."
    echo "[Trial $i] Lane is ready. Waiting 10 seconds before shutdown..."
    sleep 10

    # echo "[Trial $i] Killing all relevant processes..."
    echo "[Trial $i] Waiting for dataset node to shut down..."
    while rosnode list | grep -q "/lane_segmentation_dataset_node_6"; do
        sleep 2
    done

    # rosnode kill /lane_segmentation_dataset_node_6
    # pkill -f lane_demo_node_vel_control_figure8.py
    # pkill -f dvs_renderer
    # pkill -f spawn_random_pose.py
    # # pkill -f lane_segmentation_dataset_node_6.py
    # pkill -f monitor_and_kill_figure8.py
    # pkill -f gzserver
    # pkill -f gzclient

    pkill -f lane_demo_node_vel_control_square_smooth.py
    pkill -f dvs_renderer
    pkill -f spawn_random_pose.py
    # pkill -f lane_segmentation_dataset_node_6.py
    pkill -f monitor_and_kill_figure8.py
    pkill -f gzserver
    pkill -f gzclient

    echo "[Trial $i] All processes shut down. Proceeding to next trial..."

done

echo "All $NUM_RUNS trials completed!"
