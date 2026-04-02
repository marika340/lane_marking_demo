import os
import matplotlib.pyplot as plt

# Paths
dataset_dir = os.path.expanduser("~/catkin_ws/src/lane_marking_demo/dataset")
events_dir = os.path.join(dataset_dir, "events")
rgb_dir = os.path.join(dataset_dir, "rgb")

# Helper: Extract sorted timestamps from filenames
def extract_timestamps(directory):
    timestamps = []
    for fname in os.listdir(directory):
        if fname.endswith(".png"):
            try:
                ts = int(os.path.splitext(fname)[0])
                timestamps.append(ts)
            except ValueError:
                continue
    return sorted(timestamps)

# Load timestamps
event_ts = extract_timestamps(events_dir)
rgb_ts = extract_timestamps(rgb_dir)

# Convert from nanoseconds to seconds (optional)
event_ts_sec = [(t - event_ts[0]) * 1e-9 for t in event_ts]
rgb_ts_sec = [(t - event_ts[0]) * 1e-9 for t in rgb_ts]

# Plot
plt.figure(figsize=(12, 5))
plt.plot(event_ts_sec, [1] * len(event_ts_sec), 'b.', label="DVS Images")
plt.plot(rgb_ts_sec, [1.1] * len(rgb_ts_sec), 'r.', label="RGB Images")
plt.xlabel("Time (seconds since start)")
plt.yticks([])
plt.title("Timestamps of DVS vs RGB image saves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
