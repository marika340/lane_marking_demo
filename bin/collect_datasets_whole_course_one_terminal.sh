#!/bin/bash

NUM_RUNS=10
TRACK="figure8"  # Options: circle, figure8, square_smooth, square_tight, square_random

# ------------------------------------------------------------------
# Step 0: Start Gazebo ONCE with lanes already in the world
# ------------------------------------------------------------------
roslaunch gazebo_ros empty_world.launch \
    world_name:=$(rospack find lane_marking_demo)/worlds/${TRACK}_lanes.world \
    use_sim_time:=true &
GAZEBO_PID=$!
sleep 5

for ((i=0; i<$NUM_RUNS; i++)); do
    echo "========================"
    echo "Starting trial $i"
    echo "========================"

    if ! rosnode list | grep -q "/gazebo"; then
        echo "[WARN] Gazebo not running, restarting..."
        roslaunch gazebo_ros empty_world.launch \
            world_name:=$(rospack find lane_marking_demo)/worlds/${TRACK}_lanes.world \
            use_sim_time:=true &
        GAZEBO_PID=$!
        sleep 5
    fi

    # Step 1: Reset obstacles
    python3 ~/catkin_ws/src/lane_marking_demo/src/reset_obstacles.py --track $TRACK
    echo "Reset obstacles: done"
    sleep 2

    # Step 2: Move car
    python3 ~/catkin_ws/src/lane_marking_demo/src/move_camera_random.py --track $TRACK
    echo "Move a car: done"
    sleep 2

    # Step 3: Control node
    rosrun lane_marking_demo lane_demo_node_vel_control_${TRACK}_rand_start.py &
    CTRL_PID=$!
    sleep 2

    # Step 4: Publish camera info (one-shot, no background needed)
    rostopic pub --once --latch /lane_demo/front_cam/camera_info sensor_msgs/CameraInfo \
      "{header: {frame_id: 'front_cam'}, height: 128, width: 128, distortion_model: 'plumb_bob', D: [0, 0, 0, 0, 0], K: [128.0, 0.0, 64.0, 0.0, 128.0, 64.0, 0.0, 0.0, 1.0], R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], P: [128.0, 0.0, 64.0, 0.0, 0.0, 128.0, 64.0, 0.0, 0.0, 0.0, 1.0, 0.0]}" 
    sleep 2

    # Step 5: DVS renderer
    rosrun dvs_renderer dvs_renderer events:=/lane_demo/front_cam/events camera_info:=/lane_demo/front_cam/camera_info &
    RENDER_PID=$!
    sleep 2

    # Step 6: Dataset collector
    rosrun lane_marking_demo lane_segmentation_dataset_node_5.py &
    DATA_PID=$!
    sleep 2

    # Step 7: Monitor
    python3 ~/catkin_ws/src/lane_marking_demo/src/monitor_and_kill_efficient.py --track $TRACK &
    MONITOR_PID=$!

    echo "[Trial $i] Waiting for /lane_demo/ready signal..."
    python3 ~/catkin_ws/src/lane_marking_demo/src/wait_for_lane_ready.py
    echo "[Trial $i] Lane is ready. Continuing."

    # Wait for monitor to finish
    wait $MONITOR_PID
    echo "[Trial $i] Trial finished."

    # Kill leftover processes for this trial
    kill $CTRL_PID $RENDER_PID $DATA_PID 2>/dev/null
done

echo "All $NUM_RUNS trials completed!"
kill $GAZEBO_PID
