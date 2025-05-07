import cv2
import numpy as np
import pyrealsense2 as rs
from pathlib import Path
from tqdm import tqdm
import argparse

# ===== Constants =====
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

COLOR_DIR = "rgb_frames"
DEPTH_DIR = "depth_frames"
IMU_NPZ = "imu_data.npz"

def convert_to_dataset(bag_path: str, output_folder: str, name: str, delete_bag: bool = False):
    bag_path = Path(bag_path)
    output_root = Path(output_folder)
    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)

    color_dir = output_dir / COLOR_DIR
    depth_dir = output_dir / DEPTH_DIR
    color_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)

    color_mp4_path = output_dir / f"{name}_rgb_video.mp4"
    depth_mp4_path = output_dir / f"{name}_depth_video.mp4"

    # === Step 1: Discover streams available in the .bag ===
    ctx = rs.context()
    tmp_pipeline = rs.pipeline(ctx)
    tmp_config = rs.config()
    tmp_config.enable_device_from_file(str(bag_path), repeat_playback=False)
    tmp_profile = tmp_pipeline.start(tmp_config)
    dev = tmp_profile.get_device()
    sensors = dev.query_sensors()
    available_streams = set()
    for sensor in sensors:
        for profile in sensor.get_stream_profiles():
            available_streams.add(profile.stream_type())
    tmp_pipeline.stop()

    # === Step 2: Build config only with supported streams ===
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(str(bag_path), repeat_playback=False)

    has_color = rs.stream.color in available_streams
    has_depth = rs.stream.depth in available_streams
    has_accel = rs.stream.accel in available_streams
    has_gyro = rs.stream.gyro in available_streams

    if has_color:
        config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FPS)
    if has_depth:
        config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, FPS)
    if has_accel:
        config.enable_stream(rs.stream.accel)
    if has_gyro:
        config.enable_stream(rs.stream.gyro)

    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    # === Step 3: Set up writers ===
    color_writer = None
    depth_writer = None
    if has_color:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        color_writer = cv2.VideoWriter(str(color_mp4_path), fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
    if has_depth:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        depth_writer = cv2.VideoWriter(str(depth_mp4_path), fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

    imu_data = {
        "accel": [],
        "gyro": []
    }

    frame_count = 0
    try:
        duration_ms = playback.get_duration().total_seconds() * 1000
        estimated_frames = int(duration_ms / (1000 / FPS)) if duration_ms > 0 else 1000
        pbar = tqdm(total=estimated_frames, desc=f"[{name}] Processing frames")

        while True:
            frames = pipeline.wait_for_frames(timeout_ms=3000)
            pbar.update(1)
            frame_count += 1

            # Color
            if has_color:
                color_frame = frames.get_color_frame()
                if color_frame:
                    color_img = np.asanyarray(color_frame.get_data())
                    cv2.imwrite(str(color_dir / f"frame_{frame_count:06d}.jpg"), color_img)
                    color_writer.write(color_img)

            # Depth
            if has_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_img = np.asanyarray(depth_frame.get_data())
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET
                    )
                    cv2.imwrite(str(depth_dir / f"frame_{frame_count:06d}.jpg"), depth_colormap)
                    depth_writer.write(depth_colormap)

            # IMU
            for frame in frames:
                if frame.is_motion_frame():
                    data = frame.as_motion_frame().get_motion_data()
                    ts = frame.get_timestamp()
                    if frame.get_profile().stream_type() == rs.stream.accel:
                        imu_data["accel"].append((ts, data.x, data.y, data.z))
                    elif frame.get_profile().stream_type() == rs.stream.gyro:
                        imu_data["gyro"].append((ts, data.x, data.y, data.z))

    except RuntimeError:
        print("[INFO] End of bag or timeout.")
    finally:
        pipeline.stop()
        pbar.close()
        if color_writer:
            color_writer.release()
        if depth_writer:
            depth_writer.release()

        if has_accel or has_gyro:
            np.savez(
                output_dir / IMU_NPZ,
                accel=np.array(imu_data["accel"]),
                gyro=np.array(imu_data["gyro"])
            )

        if delete_bag:
            try:
                bag_path.unlink()
                print(f"[CLEANUP] Deleted original bag: {bag_path}")
            except Exception as e:
                print(f"[WARNING] Failed to delete bag file: {e}")


        print(f"[DONE] Saved to {output_dir}")


# ===== CLI Entry Point =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag_path", type=str, required=True, help="Path to the .bag file")
    parser.add_argument("--output_folder", type=str, required=True, help="Root output folder (e.g., test/)")
    parser.add_argument("--name", type=str, required=True, help="Name of this recording (used for subdir and video prefix)")
    parser.add_argument("--delete_bag", action="store_true", help="Delete the original .bag file after conversion")
    args = parser.parse_args()

    convert_to_dataset(args.bag_path, args.output_folder, args.name, delete_bag=args.delete_bag)

