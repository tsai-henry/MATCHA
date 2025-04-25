import time
import argparse
import cv2 

import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from teleop.robot_control.episode_loader import EpisodeLoader
from teleop.robot_control.robot_arm import H1_2_ArmController

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode_path', type=str)
    parser.add_argument('--frequency', type=int, default=30.0, help="replay data's frequency")

    args = parser.parse_args()

    ep_path = args.episode_path
    ep_loader = EpisodeLoader(ep_path)

    arm_ctrl = H1_2_ArmController()

    try:
        user_input = input("Please enter the start signal (enter 'r' to start the subsequent program):\n")

        if user_input.lower() == 'r':
            arm_ctrl.speed_gradual_max()

            idx = 0
            for i in range(len(ep_loader)):
                start_time = time.time()
                data_obj = ep_loader[i]

                left_arm_action_qpos = data_obj["left_arm_action_qpos"]
                right_arm_action_qpos = data_obj["right_arm_action_qpos"]
                qpos = left_arm_action_qpos + right_arm_action_qpos

                torques = [0.0] * len(qpos) # replay with just position control
                arm_ctrl.ctrl_dual_arm(qpos, torques)

                idx += 1

                curr_time = time.time()
                time_elapsed = curr_time - start_time
                sleep_time = max(0, (1.0 / args.frequency) - time_elapsed)
                time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("KeyboardInterrupt, safely exiting...")
    finally:
        arm_ctrl.ctrl_dual_arm_go_home()
        exit(0)
    