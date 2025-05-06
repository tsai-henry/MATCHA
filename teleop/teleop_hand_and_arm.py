import numpy as np
import time
import argparse
import cv2
from multiprocessing import shared_memory, Array, Lock
import threading

import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from teleop.open_television.tv_wrapper import TeleVisionWrapper

from teleop.robot_control.robot_arm import H1_2_ArmController
from teleop.robot_control.robot_arm_ik import H1_2_ArmIK
from teleop.robot_control.robot_hand_inspire_rh56dftp import InspireControllerRH56DFTP as InspireController
from teleop.image_server.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_dir', type = str, default = './utils/data', help = 'path to save data')
    parser.add_argument('--frequency', type = int, default = 30.0, help = 'save data\'s frequency')

    parser.add_argument('--record', action = 'store_true', help = 'Save data or not')
    parser.add_argument('--no-record', dest = 'record', action = 'store_false', help = 'Do not save data')
    parser.set_defaults(record = False)

    parser.add_argument('--arm', type=str, choices=['H1_2'], default='H1_2', help='Select arm controller')
    parser.add_argument('--hand', type=str, choices=['inspire'], default='inspire', help='Select hand controller')

    args = parser.parse_args()
    print(f"args:{args}\n")

    # image client: img_config should be the same as the configuration in image_server.py (of Robot's development computing unit)
    img_config = {
        'fps': 30,                                                          # frame per second
        'head_camera_type': 'realsense',                                  # opencv or realsense
        'head_camera_image_shape': [480, 640],                            # Head camera resolution  [height, width]
        'head_camera_id_numbers': ["332322073268"]                                    # '/dev/video0' (opencv)
    }
    ASPECT_RATIO_THRESHOLD = 2.0 # If the aspect ratio exceeds this value, it is considered binocular
    if len(img_config['head_camera_id_numbers']) > 1 or (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        BINOCULAR = True
    else:
        BINOCULAR = False
    if 'wrist_camera_type' in img_config:
        WRIST = True
    else:
        WRIST = False
    
    if BINOCULAR and not (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1] * 2, 3)
    else:
        tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)

    tv_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(tv_img_shape) * np.uint8().itemsize)
    tv_img_array = np.ndarray(tv_img_shape, dtype = np.uint8, buffer = tv_img_shm.buf)

    if WRIST:
        wrist_img_shape = (img_config['wrist_camera_image_shape'][0], img_config['wrist_camera_image_shape'][1] * 2, 3)
        wrist_img_shm = shared_memory.SharedMemory(create = True, size = np.prod(wrist_img_shape) * np.uint8().itemsize)
        wrist_img_array = np.ndarray(wrist_img_shape, dtype = np.uint8, buffer = wrist_img_shm.buf)
        img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name, 
                                 wrist_img_shape = wrist_img_shape, wrist_img_shm_name = wrist_img_shm.name)
    else:
        img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = tv_img_shm.name)

    image_receive_thread = threading.Thread(target = img_client.receive_process, daemon = True)
    image_receive_thread.daemon = True
    image_receive_thread.start()

    # television: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
    tv_wrapper = TeleVisionWrapper(BINOCULAR, tv_img_shape, tv_img_shm.name)

    # arm
    if args.arm == 'H1_2':
        arm_ctrl = H1_2_ArmController()
        arm_ik = H1_2_ArmIK()
    else:
        raise ValueError(f"Unsupported arm controller: {args.arm}")

    if args.hand == "inspire":
        left_hand_array = Array('d', 75, lock = True)          # [input]
        right_hand_array = Array('d', 75, lock = True)         # [input]
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
        dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
        dual_hand_touch_array = Array('d', 2124, lock = False)   # [output] current left, right hand touch(2124) data.

        hand_ctrl = InspireController(left_hand_array, right_hand_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, dual_hand_touch_array)
    else:
        raise ValueError(f"Unsupported hand controller: {args.hand}")
    
    if args.record:
        recorder = EpisodeWriter(task_dir = args.task_dir, frequency = args.frequency, rerun_log = True)
        recording = False
        
    try:
        user_input = input("Please enter the start signal (enter 'r' to start the subsequent program):\n")
        if user_input.lower() == 'r':
            arm_ctrl.speed_gradual_max()

            running = True
            while running:
                start_time = time.time()
                head_rmat, left_wrist, right_wrist, left_hand, right_hand = tv_wrapper.get_data()

                # send hand skeleton data to hand_ctrl.control_process
                if args.hand:
                    left_hand_array[:] = left_hand.flatten()
                    right_hand_array[:] = right_hand.flatten()

                # get current state data.
                current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
                current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

                # solve ik using motor data and wrist pose, then use ik results to control arms.
                sol_q, sol_tauff  = arm_ik.solve_ik(left_wrist, right_wrist, current_lr_arm_q, current_lr_arm_dq)
                sol_tauff = np.zeros_like(sol_tauff) # no torque control, since policy uses only pos control
                arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)

                tv_resized_image = cv2.resize(tv_img_array, (tv_img_shape[1] // 2, tv_img_shape[0] // 2))
                cv2.imshow("record image", tv_resized_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    running = False
                elif key == ord('s') and args.record:
                    recording = not recording # state flipping
                    if recording:
                        if not recorder.create_episode():
                            recording = False
                    else:
                        recorder.save_episode()

                # record data
                if args.record:
                    if args.hand == "inspire":
                        left_hand_state = dual_hand_state_array[:6]
                        right_hand_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        left_hand_touch = dual_hand_touch_array[:1062]
                        right_hand_touch = dual_hand_touch_array[-1062:]
                    else:
                        pass
                    # head image
                    current_tv_image = tv_img_array.copy()

                    # wrist image
                    if WRIST:
                        current_wrist_image = wrist_img_array.copy()

                    # arm state and action
                    left_arm_state  = current_lr_arm_q[:7]
                    right_arm_state = current_lr_arm_q[-7:]
                    left_arm_action = sol_q[:7]
                    right_arm_action = sol_q[-7:]

                    left_arm_torque = sol_tauff[:7]
                    right_arm_torque = sol_tauff[-7:]

                    if recording:
                        colors = {}
                        depths = {}
                        if BINOCULAR:
                            colors[f"color_{0}"] = current_tv_image[:, :tv_img_shape[1]//2]
                            colors[f"color_{1}"] = current_tv_image[:, tv_img_shape[1]//2:]
                            if WRIST:
                                colors[f"color_{2}"] = current_wrist_image[:, :wrist_img_shape[1]//2]
                                colors[f"color_{3}"] = current_wrist_image[:, wrist_img_shape[1]//2:]
                        else:
                            colors[f"color_{0}"] = current_tv_image
                            if WRIST:
                                colors[f"color_{1}"] = current_wrist_image[:, :wrist_img_shape[1]//2]
                                colors[f"color_{2}"] = current_wrist_image[:, wrist_img_shape[1]//2:]
                        states = {
                            "left_arm": {                                                                    
                                "dof_angles":   left_arm_state.tolist(),              
                            }, 
                            "right_arm": {                                                                    
                                "dof_angles":   right_arm_state.tolist(),                            
                            },                        
                            "left_hand": {                                                                    
                                "dof_angles":   left_hand_state,
                                "tactiles": hand_ctrl._unflatten_touch_arr(left_hand_touch)                            
                            }, 
                            "right_hand": {                                                                    
                                "dof_angles":   right_hand_state,
                                "tactiles": hand_ctrl._unflatten_touch_arr(right_hand_touch)
                            }, 
                        }
                        actions = {
                            "left_arm": {                                   
                                "dof_angles":   left_arm_action.tolist(),       
                                "dof_torques": left_arm_torque.tolist(),      
                            }, 
                            "right_arm": {                                   
                                "dof_angles":   right_arm_action.tolist(),       
                                "dof_torques": right_arm_torque.tolist(),       
                            },                         
                            "left_hand": {                                   
                                "dof_angles":   left_hand_action,   
                            }, 
                            "right_hand": {                                   
                                "dof_angles":   right_hand_action,
                            }, 
                        }
                        recorder.add_item(colors=colors, states=states, actions=actions)

                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / float(args.frequency)) - time_elapsed)
                time.sleep(sleep_time)
    except Exception as e:
        print(e)
        print("Error, safely exiting...")
    finally:
        arm_ctrl.ctrl_dual_arm_go_home()
        tv_img_shm.unlink()
        tv_img_shm.close()
        if WRIST:
            wrist_img_shm.unlink()
            wrist_img_shm.close()
        if args.record:
            recorder.close()
        print("Finally, exiting program...")
        exit(0)