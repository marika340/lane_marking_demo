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
    gnome-terminal -- bash -c "
        python3 ~/catkin_ws/src/lane_marking_demo/src/generate_random_world.py --track figure8;
        exec bash
    "
    # gnome-terminal -- bash -c "
    #     python3 ~/catkin_ws/src/lane_marking_demo/src/generate_random_world.py --track square_smooth;
    #     exec bash
    # "
    sleep 2

    # Step 2: Launch Gazebo
    gnome-terminal -- bash -c "
        roslaunch gazebo_ros empty_world.launch world_name:=$(rospack find lane_marking_demo)/worlds/random_trial.world use_sim_time:=true;
        exec bash
    "
    sleep 5

    # Step 3: Spawn random vehicle pose
    gnome-terminal -- bash -c "
        cd ~/catkin_ws && python3 ./src/lane_marking_demo/src/spawn_random_pose_random_part.py;
        exec bash
    "
    sleep 2

    # Step 4: Start lane drawing + control node
    # gnome-terminal -- bash -c "
    #     rosrun lane_marking_demo lane_demo_node_vel_control_circle.py;
    #     exec bash
    # "
    gnome-terminal -- bash -c "
        rosrun lane_marking_demo lane_demo_node_vel_control_figure8_rand_start.py;
        exec bash
    "
    # gnome-terminal -- bash -c "
    #     rosrun lane_marking_demo lane_demo_node_vel_control_figure8_opposite.py;
    #     exec bash
    # "
    # gnome-terminal -- bash -c "
    #     rosrun lane_marking_demo lane_demo_node_vel_control_square_smooth.py;
    #     exec bash
    # "
    # gnome-terminal -- bash -c "
    #     rosrun lane_marking_demo lane_demo_node_vel_control_square_random.py;
    #     exec bash
    # "
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
        rosrun lane_marking_demo lane_segmentation_dataset_node_5.py;
        exec bash
    "

    # Step 8: Monitor when vehicle returns to initial pose after ready
    # circle, square
    # gnome-terminal -- bash -c "
    #     echo '[Trial $i] Monitoring vehicle return...';
    #     python3 ~/catkin_ws/src/lane_marking_demo/src/monitor_and_kill.py;
    #     exec bash
    # "

    # figure8
    gnome-terminal -- bash -c "
        echo '[Trial $i] Monitoring vehicle return...';
        python3 ~/catkin_ws/src/lane_marking_demo/src/monitor_and_kill_figure8.py;
        exec bash
    "

    echo "[Trial $i] Waiting for /lane_demo/ready signal..."
    python3 ~/catkin_ws/src/lane_marking_demo/src/wait_for_lane_ready.py
    echo "[Trial $i] Lane is ready. Continuing."

    # circle, square
    # echo "[Trial $i] Waiting for monitor_and_kill.py to finish..."
    # while pgrep -f monitor_and_kill.py > /dev/null; do
    #     sleep 2
    # done

    # figure8
    echo "[Trial $i] Waiting for monitor_and_kill_figure8.py to finish..."
    while pgrep -f monitor_and_kill_figure8.py > /dev/null; do
        sleep 2
    done

    # echo "[Trial $i] Waiting for all processes to shut down cleanly..."
    # Wait until all critical processes are gone
    # circle
    # while pgrep -f gzserver > /dev/null || \
    #     pgrep -f gzclient > /dev/null || \
    #     pgrep -f lane_demo_node_vel_control_circle.py > /dev/null || \
    #     pgrep -f dvs_renderer > /dev/null || \
    #     pgrep -f spawn_random_pose.py > /dev/null || \
    #     pgrep -f lane_segmentation_dataset_node_5.py > /dev/null || \
    #     rosnode list | grep -q "lane_segmentation_dataset_node"; do
    #     sleep 2
    # done

    # figure8
    while pgrep -f gzserver > /dev/null || \
        pgrep -f gzclient > /dev/null || \
        pgrep -f lane_demo_node_vel_control_figure8.py > /dev/null || \
        pgrep -f dvs_renderer > /dev/null || \
        pgrep -f spawn_random_pose.py > /dev/null || \
        pgrep -f lane_segmentation_dataset_node_5.py > /dev/null || \
        rosnode list | grep -q "lane_segmentation_dataset_node"; do
        sleep 2
    done

    # figure8 opposite
    # while pgrep -f gzserver > /dev/null || \
    #     pgrep -f gzclient > /dev/null || \
    #     pgrep -f lane_demo_node_vel_control_figure8_opposite.py > /dev/null || \
    #     pgrep -f dvs_renderer > /dev/null || \
    #     pgrep -f spawn_random_pose.py > /dev/null || \
    #     pgrep -f lane_segmentation_dataset_node_5.py > /dev/null || \
    #     rosnode list | grep -q "lane_segmentation_dataset_node"; do
    #     sleep 2
    # done

    # square_smooth
    # while pgrep -f gzserver > /dev/null || \
    #     pgrep -f gzclient > /dev/null || \
    #     pgrep -f lane_demo_node_vel_control_square_smooth.py > /dev/null || \
    #     pgrep -f dvs_renderer > /dev/null || \
    #     pgrep -f spawn_random_pose.py > /dev/null || \
    #     pgrep -f lane_segmentation_dataset_node_5.py > /dev/null || \
    #     rosnode list | grep -q "lane_segmentation_dataset_node"; do
    #     sleep 2
    # done

    # square_random
    # while pgrep -f gzserver > /dev/null || \
    #     pgrep -f gzclient > /dev/null || \
    #     pgrep -f lane_demo_node_vel_control_square_random.py > /dev/null || \
    #     pgrep -f dvs_renderer > /dev/null || \
    #     pgrep -f spawn_random_pose_random_part.py > /dev/null || \
    #     pgrep -f lane_segmentation_dataset_node_5.py > /dev/null || \
    #     rosnode list | grep -q "lane_segmentation_dataset_node"; do
    #     sleep 2
    # done

    echo "[Trial $i] All processes shut down. Proceeding to next trial..."

done

echo "All $NUM_RUNS trials completed!"
