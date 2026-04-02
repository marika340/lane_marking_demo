#!/bin/bash
set -e

CONFIG_FILE="$(rospack find lane_marking_demo)/config/config2.json"

# --- Cleanup function ---
cleanup() {
    echo "Cleaning up..."
    killall -9 gzserver gzclient 2>/dev/null || true
    kill $GAZEBO_PID $CTRL_PID $RENDER_PID $DATA_PID $MONITOR_PID 2>/dev/null || true
}
trap cleanup EXIT   # Call cleanup on script exit (Ctrl+C, error, or normal exit)


# Read values from JSON
NUM_RUNS=$(jq '.NUM_RUNS' $CONFIG_FILE)
TRACK=$(jq -r '.TRACK' $CONFIG_FILE)
CAMERA_RANGE=$(jq '.camera_range' $CONFIG_FILE)
CAMERA_HEIGHT=$(jq '.camera_height' $CONFIG_FILE)
DURATION_MIN=$(jq '.duration_min' $CONFIG_FILE)
DURATION_MAX=$(jq '.duration_max' $CONFIG_FILE)
DEPART_THRESH=$(jq '.depart_thresh' $CONFIG_FILE)
TOLERANCE=$(jq '.tolerance' $CONFIG_FILE)

echo "Loaded configuration:"
echo "  NUM_RUNS     = $NUM_RUNS"
echo "  TRACK        = $TRACK"
echo "  CAMERA_RANGE = $CAMERA_RANGE"
echo "  CAMERA_HEIGHT= $CAMERA_HEIGHT"
echo "  DURATION_MIN = $DURATION_MIN"
echo "  DURATION_MAX = $DURATION_MAX"
echo "  DEPART_THRESH= $DEPART_THRESH"
echo "  TOLERANCE    = $TOLERANCE"

# ------------------------------------------------------------------
# Step 0: Start Gazebo ONCE with lanes already in the world
# ------------------------------------------------------------------
roslaunch gazebo_ros empty_world.launch \
    world_name:=$(rospack find lane_marking_demo)/worlds/${TRACK}_lanes.world \
    use_sim_time:=true &
GAZEBO_PID=$!
sleep 5
    
# Step 4: Publish camera info
rostopic pub --once --latch /lane_demo/front_cam/camera_info sensor_msgs/CameraInfo \
    "{header: {frame_id: 'front_cam'}, height: 128, width: 128, distortion_model: 'plumb_bob', D: [0, 0, 0, 0, 0], K: [128.0, 0.0, 64.0, 0.0, 128.0, 64.0, 0.0, 0.0, 1.0], R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], P: [128.0, 0.0, 64.0, 0.0, 0.0, 128.0, 64.0, 0.0, 0.0, 0.0, 1.0, 0.0]}" 

for ((i=0; i<$NUM_RUNS; i++)); do
    echo "========================"
    echo "Starting trial $i"
    echo "========================"

    # Pick a random duration within [min, max]
    DURATION=$(shuf -i ${DURATION_MIN}-${DURATION_MAX} -n 1)
    echo "[Trial $i] Selected duration: $DURATION seconds"

    if ! rosnode list | grep -q "/gazebo"; then
        echo "[WARN] Gazebo not running, restarting..."
        roslaunch gazebo_ros empty_world.launch \
            world_name:=$(rospack find lane_marking_demo)/worlds/${TRACK}_lanes.world \
            use_sim_time:=true &
        GAZEBO_PID=$!
        sleep 5
    fi

    # Step 1: Reset obstacles
    if ((i%10 == 0)); then
        python3 ~/catkin_ws/src/lane_marking_demo/src/reset_obstacles.py --track $TRACK
        echo "Reset obstacles: done"
        # sleep 2
    fi

    # Step 2: Move car
    python3 ~/catkin_ws/src/lane_marking_demo/src/move_camera_random.py \
        --track $TRACK \
        --range $CAMERA_RANGE \
        --height $CAMERA_HEIGHT
    echo "Move a car: done"
    # sleep 2

    # Step 3: Control node
    rosrun lane_marking_demo lane_demo_node_vel_control_${TRACK}_rand_start.py &
    CTRL_PID=$!
    sleep 2

    # Step 5: DVS renderer
    rosrun dvs_renderer dvs_renderer events:=/lane_demo/front_cam/events camera_info:=/lane_demo/front_cam/camera_info &
    RENDER_PID=$!
    sleep 2

    # Step 6: Dataset collector
    rosrun lane_marking_demo lane_segmentation_dataset_node_5.py --track $TRACK &
    DATA_PID=$!
    sleep 2


    # Step 7: Monitor with random duration
    python3 ~/catkin_ws/src/lane_marking_demo/src/monitor_and_kill_timed.py \
        --track $TRACK \
        --duration $DURATION \
        --depart_thresh $DEPART_THRESH \
        --tolerance $TOLERANCE &
    MONITOR_PID=$!

    echo "[Trial $i] Waiting for /lane_demo/ready signal..."
    python3 ~/catkin_ws/src/lane_marking_demo/src/wait_for_lane_ready.py
    echo "[Trial $i] Lane is ready. Continuing."

    wait $MONITOR_PID
    echo "[Trial $i] Trial finished."

    kill $CTRL_PID $RENDER_PID $DATA_PID 2>/dev/null
    sleep 5
done

echo "All $NUM_RUNS trials completed!"
kill $GAZEBO_PID
