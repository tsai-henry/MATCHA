<h1 align="center">
MATCHA: Mobility Augmenting Teleoperation-Controlled Humanoid Arm 🍵
</h1>

![Experimental Setup](./MATCHA%20Water%20Bottle%20Grasping%20Demo.png)
<p align="center">
  <a href="https://drive.google.com/drive/folders/1Snr74SE7jA_POdz03ILZVOsoFncml9S2?usp=sharing"><strong>Download Grasping Dataset</strong></a>
</p>
<p align="center">
  <a href="./MATCHA%20Paper.pdf"><strong>Paper PDF</strong></a>
</p>


## Overview

MATCHA is a teleoperation framework designed to enable individuals with hand disabilities (specifically missing fingers or restricted motion) to control humanoid robots with multi-fingered hands effectively. By using an Apple Vision Pro VR headset for hand tracking, the system maps limited human inputs (thumb and index finger) to actuate a pair of 5-fingered Inspire dexterous hands on the Unitree H1-2 humanoid.

This repository contains the updated hand controller drivers, control algorithms, and dataset collection tools built on top of the open-source [Unitree XR Teleoperate](https://github.com/unitreerobotics/avp_teleoperate) project.

## Hardware Setup

The system is designed for the following hardware configuration:

*   **Humanoid Robot:** Unitree H1-2 (with 6-DOF arms)
*   **Robotic Hands:** Inspire RH56DFTP
    *   5 fingers, 6-DOF actuation
    *   Integrated tactile sensors (pressure mapping)
*   **Teleoperation Interface:** Apple Vision Pro (AVP)
    *   Used for VR hand tracking and command streaming
*   **Perception:** Intel RealSense D435i (RGB-D Camera)
*   **Compute:** Razer 15 Laptop (Host)
*   **Network:** Dedicated low-latency router connecting AVP, Robot, and Host.

## Software Architecture

We extend the Unitree `avp_teleoperate` stack to support:
1.  **Inspire Hand Integration:** Custom drivers to actuate the RH56DFTP hands and read tactile sensor data.
2.  **Augmented Control Strategies:** Algorithms to control unactuated "support" fingers (Middle, Ring, Pinky) based on the active "main" fingers (Thumb, Index).
3.  **Data Collection:** Tools to record RGB-D footage and telemetry for grasp analysis.

## Control Strategies

To accommodate users with limited hand function, we implemented two controllers that automate the movement of the unactuated fingers:

### 1. Pose Matching Controller
This method treats the unactuated fingers as "mimics" of the index finger.
*   **Logic:** Copies the joint angle of the source finger (Index) to the Middle, Ring, and Pinky fingers.
*   **Use Case:** Effective for power grasps where all fingers curl simultaneously.
*   **Success Rate:** 58% (based on experimental trials).

### 2. Pressure Matching Controller (Force Matching)
This method uses tactile feedback to adjust the unactuated fingers dynamically.
*   **Logic:** The support fingers adjust their curl to match the pressure experienced by the index finger.
    *   If `Pressure_Support < Pressure_Index`: Curl inward (increase grip).
    *   If `Pressure_Support > Pressure_Index`: Extend outward (release grip).
*   **Parameters:** Includes a deadband threshold ($\epsilon$) to filter sensor noise and a step size ($\delta$) for movement.
*   **Success Rate:** 62% (based on experimental trials).

## Installation

Please follow the installation instructions in the [Unitree XR Teleoperate](https://github.com/unitreerobotics/avp_teleoperate) repository.

## Usage

### 1. Configuration
*   **SSL Certificates:** The Vuer interface requires HTTPS. Generate `cert.pem` and `key.pem` in the project root:
    ```bash
    # Example using mkcert
    mkcert -key-file key.pem -cert-file cert.pem 0.0.0.0 localhost 127.0.0.1
    ```
*   **IP Address:** Verify the robot's IP address in `teleop/image_server/image_client.py` (Default: `192.168.123.164`).

### 2. Teleoperation
1.  **Start the Robot:** Place the Unitree H1-2 in debug mode. Ensure it is suspended or stable.
2.  **Launch VR Interface:** 
    *   On the Apple Vision Pro, navigate to `https://<HOST_IP>:8012` (allow self-signed certificates if prompted).
3.  **Run Teleoperation & Data Collection:**
    Execute the main controller script. This handles robot control, VR streaming, and optional data recording.

    ```bash
    python teleop/teleop_hand_and_arm.py --record --name my_experiment
    ```

    **Arguments:**
    *   `--record`: Enable data collection (RGB-D, proprioception, VR inputs).
    *   `--name`: Name tag for the recording (default: timestamp).
    *   `--task_dir`: Directory to save data (default: `./utils/data`).
    *   `--frequency`: Recording frequency (default: 30Hz).

## Results

We conducted 100 pick-item grasping trials across 10 everyday objects (e.g., pear, water bottle, stapler).

| Controller | Success Rate | Key Observations |
| :--- | :--- | :--- |
| **Pose Matching** | 58% | Reliable for simple shapes; struggles with fragile items. |
| **Force Matching** | 62% | Better adaptation; prone to jitter due to sensor noise. |

## Acknowledgments

*   **Professor Shankar Sastry** for providing resources and teaching fundamentals.
*   **Anirudh Pai** for working on the Inspire hand controller and networking setup.
*   **Haoru Xue** for debugging, guidance, and mentorship.
*   **Unitree Robotics** for the base `avp_teleoperate` library.
*   **UC Berkeley EECS Instructional Support Group and course staff** for helping make all of this possible.
