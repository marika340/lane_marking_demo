#!/bin/bash

NUM_RUNS=10
TRACK="figure8"  # Options: circle, figure8, square_smooth, square_tight, square_random

# ------------------------------------------------------------------
# Step 0: Start Gazebo ONCE with lanes already in the world
# ------------------------------------------------------------------
gnome-terminal -- bash -c "
    roslaunch gazebo_ros empty_world.launch \
        world_name:=$(rospack find lane_marking_demo)/worlds/${TRACK}_lanes.world \
        use_sim_time:=true;
    exec bash
"
sleep 5


for ((i=0; i<$NUM_RUNS; i++)); do
    echo "========================"
    echo "Starting trial $i"
    echo "========================"

    if ! rosnode list | grep -q "/gazebo"; then
        echo "[WARN] Gazebo not running, restarting..."
        roslaunch gazebo_ros empty_world.launch world_name:=$(rospack find lane_marking_demo)/worlds/${TRACK}_lanes.world use_sim_time:=true &
        sleep 5
    fi

    # ------------------------------------------------------------------
    # Step 1: Reset obstacles (delete + respawn)
    # ------------------------------------------------------------------
    # rosservice call /gazebo/pause_physics
    python3 ~/catkin_ws/src/lane_marking_demo/src/reset_obstacles.py --track $TRACK
    echo "Reset obstacles: done"
    # rosservice call /gazebo/unpause_physics
    sleep 2

    # ------------------------------------------------------------------
    # Step 2: Move car to a random valid pose
    # ------------------------------------------------------------------
    python3 ~/catkin_ws/src/lane_marking_demo/src/move_camera_random.py --track $TRACK
    echo "Move a car: done"
    sleep 2


    # ------------------------------------------------------------------
    # Step 3: Start lane following / control node
    # ------------------------------------------------------------------
    gnome-terminal -- bash -c "
        rosrun lane_marking_demo lane_demo_node_vel_control_${TRACK}_rand_start.py;
        exec bash
    "
    sleep 2
    # gnome-terminal -- bash -c "
    #     rosrun lane_marking_demo lane_demo_node_vel_control_circle.py;
    #     exec bash
    # "
    # gnome-terminal -- bash -c "
    #     rosrun lane_marking_demo lane_demo_node_vel_control_figure8_rand_start.py;
    #     exec bash
    # "
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

    # ------------------------------------------------------------------
    # Step 4: Publish camera info
    # ------------------------------------------------------------------
    gnome-terminal -- bash -c "
        rostopic pub --once --latch /lane_demo/front_cam/camera_info sensor_msgs/CameraInfo \"{header: {frame_id: 'front_cam'}, height: 128, width: 128, distortion_model: 'plumb_bob', D: [0, 0, 0, 0, 0], K: [128.0, 0.0, 64.0, 0.0, 128.0, 64.0, 0.0, 0.0, 1.0], R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], P: [128.0, 0.0, 64.0, 0.0, 0.0, 128.0, 64.0, 0.0, 0.0, 0.0, 1.0, 0.0]}\";
        exec bash
    "
    sleep 2

    # ------------------------------------------------------------------
    # Step 5: Start DVS renderer
    # ------------------------------------------------------------------
    gnome-terminal -- bash -c "
        rosrun dvs_renderer dvs_renderer events:=/lane_demo/front_cam/events camera_info:=/lane_demo/front_cam/camera_info;
        exec bash
    "
    sleep 2

    # ------------------------------------------------------------------
    # Step 6: Start dataset collector
    # ------------------------------------------------------------------
    gnome-terminal -- bash -c "
        rosrun lane_marking_demo lane_segmentation_dataset_node_5.py;
        exec bash
    "

    # ------------------------------------------------------------------
    # Step 7: Monitor and wait for run completion
    # ------------------------------------------------------------------
    # circle, square
    gnome-terminal -- bash -c "
        echo '[Trial $i] Monitoring vehicle return...';
        python3 ~/catkin_ws/src/lane_marking_demo/src/monitor_and_kill_efficient.py --track $TRACK;
        exec bash
    "

    # # figure8
    # gnome-terminal -- bash -c "
    #     echo '[Trial $i] Monitoring vehicle return...';
    #     python3 ~/catkin_ws/src/lane_marking_demo/src/monitor_and_kill_figure8.py;
    #     exec bash
    # "

    echo "[Trial $i] Waiting for /lane_demo/ready signal..."
    python3 ~/catkin_ws/src/lane_marking_demo/src/wait_for_lane_ready.py
    echo "[Trial $i] Lane is ready. Continuing."

    # circle, square
    echo "[Trial $i] Waiting for monitor_and_kill_efficient.py to finish..."
    while pgrep -f monitor_and_kill_efficient.py > /dev/null; do
        sleep 2
    done

    echo "[Trial $i] Trial finished."

    # figure8
    # echo "[Trial $i] Waiting for monitor_and_kill_figure8.py to finish..."
    # while pgrep -f monitor_and_kill_figure8.py > /dev/null; do
    #     sleep 2
    # done

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
    # while pgrep -f gzserver > /dev/null || \
    #     pgrep -f gzclient > /dev/null || \
    #     pgrep -f lane_demo_node_vel_control_figure8.py > /dev/null || \
    #     pgrep -f dvs_renderer > /dev/null || \
    #     pgrep -f spawn_random_pose.py > /dev/null || \
    #     pgrep -f lane_segmentation_dataset_node_5.py > /dev/null || \
    #     rosnode list | grep -q "lane_segmentation_dataset_node"; do
    #     sleep 2
    # done

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

    # echo "[Trial $i] All processes shut down. Proceeding to next trial..."

done

echo "All $NUM_RUNS trials completed!"
