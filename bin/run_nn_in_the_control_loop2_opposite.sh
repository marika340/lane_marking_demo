#!/usr/bin/env bash
set -euo pipefail

# ---- DIAGNOSTICS ----
echo "[diag] SHELL=$SHELL"
echo "[diag] whoami=$(whoami)"
echo "[diag] BASH_VERSION=${BASH_VERSION:-not_bash?}"

# Never run with sudo for this script
if [ "${SUDO_USER-}" ]; then
  echo "[error] Don't run this script with sudo. Exiting."; exit 1
fi

# Source ROS and your workspace BEFORE any 'rospack' or ROS commands
ROS_SETUP=${ROS_SETUP:-/opt/ros/noetic/setup.bash}
WS=${WS:-$HOME/catkin_ws}
WS_SETUP="$WS/devel/setup.bash"

if [ ! -f "$ROS_SETUP" ]; then
  echo "[error] Can't find $ROS_SETUP"; exit 1
fi
source "$ROS_SETUP"

if [ ! -f "$WS_SETUP" ]; then
  echo "[error] Can't find $WS_SETUP (did you catkin_make?)"; exit 1
fi
source "$WS_SETUP"

echo "[diag] ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH"
echo "[diag] rospack path=$(command -v rospack || echo 'not found')"

# PROVE the package is visible *inside this script*
rospack find lane_marking_demo || { echo "[error] rospack can't see lane_marking_demo"; exit 1; }

# If you use rospack to build paths, do it AFTER sourcing:
WORLD_PATH="$(rospack find lane_marking_demo)/worlds/square_lanes.world"
echo "[diag] WORLD_PATH=$WORLD_PATH"

# ---- your real commands go below ----


########################
# Config (edit as needed)
########################
ROS_SETUP=/opt/ros/noetic/setup.bash
WS=${WS:-$HOME/catkin_ws}
WS_SETUP="$WS/devel/setup.bash"

# Source environments
[ -f "$ROS_SETUP" ] && source "$ROS_SETUP"
[ -f "$WS_SETUP" ]  && source "$WS_SETUP"

# (Optional) sanity check
echo "[DEBUG] finding package lane_marking_demo..."
rospack find lane_marking_demo >/dev/null
echo "[DEBUG] found lane_marking_demo package."
PKG="lane_marking_demo"
WORLD="${HOME}/catkin_ws/src/lane_marking_demo/worlds/square_lanes.world"

# If your spawn script isn’t installed as a ROS node, use the path:
echo "[DEBUG] finding spawn script..."
SPAWN_SCRIPT_PATH="${SPAWN_SCRIPT_PATH:-$WS/src/$PKG/src/spawn_random_pose_opposite.py}"
USE_ROSRUN_SPAWN="${USE_ROSRUN_SPAWN:-0}"   # set to 1 if you have an entry-point to rosrun

LOGDIR="$WS/src/$PKG/logs/$(date +d%m_%d_t%H_%M_%S)"
mkdir -p "$LOGDIR"

################################
# Env
################################
if [ -f "$ROS_SETUP" ]; then source "$ROS_SETUP"; fi
if [ -f "$WS_SETUP"  ]; then source "$WS_SETUP";  fi

################################
# Helpers
################################
pids=()
names=()

start() {
  local name="$1"; shift
  echo "[START] $name"
  # start in background, redirect to log
  ("$@") &
  local pid=$!
  pids+=("$pid"); names+=("$name")
  echo "  -> $name PID=$pid (logs: $LOGDIR/${name}.*)"
}

wait_for_master() {
  echo "[WAIT] ROS master..."
  # roslaunch will start master if needed; just wait until rostopic responds
  until rostopic list >/dev/null 2>&1; do sleep 1; done
  echo "  -> master OK"
}

wait_for_topic() {
  local topic="$1"
  echo "[WAIT] topic: $topic"
  until rostopic info "$topic" >/dev/null 2>&1; do sleep 1; done
  echo "  -> $topic available"
}

cleanup() {
  echo
  echo "[CLEANUP] Stopping started processes..."
  # Try graceful shutdown first
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  # Give them a moment, then force kill leftovers
  sleep 2
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
  echo "[CLEANUP] Done."
}
trap cleanup EXIT INT TERM

################################
# 1) Gazebo (roslaunch)
################################
# --wait will wait for roscore (and avoid race)
echo "[LAUNCH] Gazebo world: $WORLD"
start "gazebo" roslaunch --wait gazebo_ros empty_world.launch \
  "world_name:=${WORLD}" use_sim_time:=true

wait_for_master

# Gazebo topics can take a moment; give a small buffer
sleep 2

################################
# 2) Spawn random pose
################################
echo "[LAUNCH] Spawn random pose"
if [ "$USE_ROSRUN_SPAWN" = "1" ]; then
  start "spawn_random_pose" rosrun lane_marking_demo spawn_random_pose.py
else
  # run the script directly (make sure it’s executable)
  chmod +x "$SPAWN_SCRIPT_PATH"
  start "spawn_random_pose" "$SPAWN_SCRIPT_PATH"
fi

################################
# 3) Mask publisher node
################################
echo "[LAUNCH] Mask publisher node"
start "mask_publisher" rosrun lane_marking_demo mask_publisher_node.py

################################
# 4) Publish CameraInfo (latched, once)
################################
# It’s fine to publish now; it’s latched and consumers can connect later.
echo "[PUB] /lane_demo/front_cam/camera_info (latched, once)"
rostopic pub --once --latch /lane_demo/front_cam/camera_info sensor_msgs/CameraInfo \
"{header: {frame_id: 'front_cam'}, height: 128, width: 128, distortion_model: 'plumb_bob', D: [0, 0, 0, 0, 0], K: [128.0, 0.0, 64.0, 0.0, 128.0, 64.0, 0.0, 0.0, 1.0], R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], P: [128.0, 0.0, 64.0, 0.0, 0.0, 128.0, 64.0, 0.0, 0.0, 0.0, 1.0, 0.0]}" \
  >"$LOGDIR/camera_info.once.out" 2>"$LOGDIR/camera_info.once.err" || true

################################
# 5) DVS renderer
################################
start "dvs_renderer" rosrun dvs_renderer dvs_renderer \
  events:=/lane_demo/front_cam/events \
  camera_info:=/lane_demo/front_cam/camera_info

# Wait until rendered topic exists before throttling it
wait_for_topic "/dvs_rendering"

################################
# 6) Throttle /dvs_rendering to 30 Hz
################################
# Note: topic_tools creates an output topic named /throttled by default.
start "throttle_dvs" rosrun topic_tools throttle messages /dvs_rendering 30.0

################################
# 7) Ackermann → cmd_vel
################################
start "ackermann_to_cmdvel" rosrun lane_marking_demo ackermann_to_cmdvel.py

################################
# 8) NN controller
################################
start "nn_controller" rosrun lane_marking_demo nn_controller_node.py

################################
# Status + foreground wait
################################
echo
echo "[OK] All processes started. Logs in: $LOGDIR"
echo "     Press Ctrl+C to stop everything."
echo
# Keep the script in the foreground until any child exits
# If any background process exits, we’ll still keep the script alive unless it errors.
# You can switch to `wait -n` if you want to exit when the first one stops.
wait
