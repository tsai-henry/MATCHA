# MATCHA: Mobility Augmenting Teleoperation-Controlled Humanoid Arm
![Experimental Setup](./MATCHA%20Water%20Bottle%20Grasping%20Demo.png)
<p align="center">
  <a href="https://drive.google.com/drive/folders/1Snr74SE7jA_POdz03ILZVOsoFncml9S2?usp=sharing"><strong>Download Grasping Dataset</strong></a>
</p>
<p align="center">
  <a href="./MATCHA%20Paper.pdf"><strong>Paper PDF</strong></a>
</p>
## Overview

MATCHA is a teleoperation framework designed to enable individuals with hand disabilities (specifically missing fingers or restricted motion) to control humanoid robots effectively. By using a Virtual Reality (VR) headset for hand tracking, the system maps limited human inputs (Thumb and Index finger) to a full 5-fingered robotic hand using novel interpolation strategies.

This repository contains the updated hand controller drivers, control algorithms, and dataset collection tools built on top of the open-source [AVP Teleoperate](https://github.com/unitreerobotics/avp_teleoperate) project.

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
1.  **Inspire Hand Integration:** Custom drivers to actuate the RH56DFTP hands and read tactile sensor data, which were not previously supported.
2.  **Augmented Control Strategies:** Algorithms to control unactuated "support" fingers (Middle, Ring, Pinky) based on the active "master" fingers (Thumb, Index).
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

1.  Clone this repository:
    ```bash
    git clone https://github.com/tsai-henry/MATCHA.git
    cd MATCHA
    ```

2.  Install dependencies (ensure you have the Unitree SDKs installed):
    ```bash
    pip install -r requirements.txt
    ```

3.  **Network Configuration:**
    *   Ensure the Apple Vision Pro and the Robot are on the same local network.
    *   Update the IP configuration in the config scripts to match your router's assignment.

## Usage

1.  **Start the Robot:** Place the Unitree H1-2 in debug mode and ensure it is suspended or stable.
2.  **Launch VR Interface:** Start the hand tracking application on the Apple Vision Pro.
3.  **Run Teleoperation:**
    Execute the main control script, specifying the desired control mode:

    ```bash
    # For Pose Matching
    python main.py --mode pose

    # For Pressure/Force Matching
    python main.py --mode force
    ```

## Results

We conducted 100 trials across 10 everyday objects (e.g., pear, water bottle, stapler).

| Controller | Success Rate | Key Observations |
| :--- | :--- | :--- |
| **Pose Matching** | 58% | Reliable for simple shapes; struggles with fragile items (dented pears). |
| **Force Matching** | 62% | Better adaptation; prone to jitter due to sensor noise. |

## Acknowledgments

*   **Unitree Robotics** for the base `avp_teleoperate` library.
*   **Trevor Darrell's Lab** for providing the Unitree H1-2 robot.
*   **Anirudh Pai** for assistance with the Inspire hand controller.
*   **Haoru Xue** and **Prof. Sastry** for debugging and guidance.
