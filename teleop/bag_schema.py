import pyrealsense2 as rs
import numpy as np
import cv2
import sys

bag_file = "/home/zekai/Documents/realsense_recordings/" + str(sys.argv[1]) + ".bag"
# Set up pipeline to read from file
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_file, repeat_playback=False)
config.enable_stream(rs.stream.color)
config.enable_stream(rs.stream.depth)

# Start pipeline
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Visualize depth as colormap
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        # Show both images side by side
        combined = np.hstack((color_image, depth_colormap))
        cv2.imshow('Color and Depth', combined)

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()