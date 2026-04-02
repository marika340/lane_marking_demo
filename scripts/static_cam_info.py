#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import CameraInfo

def main():
    rospy.init_node("static_cam_info")
    pub = rospy.Publisher(
        "/lane_demo/front_cam/camera_info",
        CameraInfo,
        queue_size=1,
        latch=True
    )

    # Fill in a minimal CameraInfo matching width=128, height=128
    cam_info = CameraInfo()
    cam_info.header.frame_id = "front_cam"
    cam_info.width  = 128
    cam_info.height = 128

    # Intrinsics can be identity-ish, since dvs_renderer only needs
    # width/height to size its event buffer. 
    cam_info.K = [1.0, 0.0, 64.0,
                  0.0, 1.0, 64.0,
                  0.0, 0.0,  1.0]
    cam_info.P = [1.0, 0.0, 64.0, 0.0,
                  0.0, 1.0, 64.0, 0.0,
                  0.0, 0.0,  1.0, 0.0]

    # Publish once (latched), then keep alive so downstream nodes can latch it
    cam_info.header.stamp = rospy.Time.now()
    pub.publish(cam_info)
    rospy.loginfo("Published static CameraInfo on /lane_demo/front_cam/camera_info")
    rospy.spin()

if __name__ == "__main__":
    main()

