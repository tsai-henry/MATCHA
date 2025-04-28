import time
import numpy as np
import threading
from multiprocessing import Process, Array, Lock
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from inspire_sdkpy import inspire_dds, inspire_hand_defaut
from teleop.robot_control.hand_retargeting import HandRetargeting, HandType

# Constants
Inspire_Num_Motors = 6
inspire_tip_indices = [4, 9, 14, 19, 24]
kTopicInspireCommandR = "rt/inspire_hand/ctrl/r"
kTopicInspireCommandL = "rt/inspire_hand/ctrl/l"
kTopicInspireStateR = "rt/inspire_hand/state/r"
kTopicInspireStateL = "rt/inspire_hand/state/l"

class InspireControllerNewGen:
    def __init__(self, left_hand_array, right_hand_array, dual_hand_data_lock=None, dual_hand_state_array=None, dual_hand_action_array=None, fps=100.0, Unit_Test=False):
        print("Initializing Inspire_Controller...")
        self.fps = fps
        self.Unit_Test = Unit_Test

        # if not self.Unit_Test:
        #     print('here')
        #     ChannelFactoryInitialize(0, None)  # Change if needed
        # else:
        #     print('here2')
        #     ChannelFactoryInitialize(0)

        # Initialize publishers
        self.left_hand_pub = ChannelPublisher(kTopicInspireCommandL, inspire_dds.inspire_hand_ctrl)
        self.right_hand_pub = ChannelPublisher(kTopicInspireCommandR, inspire_dds.inspire_hand_ctrl)
        self.left_hand_pub.Init()
        self.right_hand_pub.Init()

        # Initialize subscribers
        self.left_hand_sub = ChannelSubscriber(kTopicInspireStateL, inspire_dds.inspire_hand_state)
        self.right_hand_sub = ChannelSubscriber(kTopicInspireStateR, inspire_dds.inspire_hand_state)
        self.left_hand_sub.Init()
        self.right_hand_sub.Init()

        # Shared Arrays
        self.left_hand_state_array = Array('d', Inspire_Num_Motors, lock=True)
        self.right_hand_state_array = Array('d', Inspire_Num_Motors, lock=True)

        # External Shared Arrays
        self.shared_left_hand_array = left_hand_array
        self.shared_right_hand_array = right_hand_array
        self.dual_hand_data_lock = dual_hand_data_lock
        self.dual_hand_state_array = dual_hand_state_array
        self.dual_hand_action_array = dual_hand_action_array

        # Initialize hand retargeting
        if not self.Unit_Test:
            self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND)
        else:
            self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND_Unit_Test)

        # Control message
        self.left_cmd = inspire_hand_defaut.get_inspire_hand_ctrl()
        self.right_cmd = inspire_hand_defaut.get_inspire_hand_ctrl()
        self.left_cmd.mode = 0b0001
        self.right_cmd.mode = 0b0001

        # Start subscription thread
        self.running = True
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        # Wait until some data received
        while True:
            if any(self.right_hand_state_array):
                break
            print("[Inspire_Controller] Waiting to subscribe DDS...")
            time.sleep(0.01)

        # Start control process
        self.hand_control_process = Process(
            target=self.control_process,
            args=(self.shared_left_hand_array, self.shared_right_hand_array, self.left_hand_state_array, self.right_hand_state_array, self.dual_hand_data_lock, self.dual_hand_state_array, self.dual_hand_action_array)
        )
        self.hand_control_process.daemon = True
        self.hand_control_process.start()

        print("Inspire_Controller initialized OK.\n")

    def _subscribe_hand_state(self):
        while self.running:
            left_msg = self.left_hand_sub.Read()
            if left_msg is not None:
                for idx in range(Inspire_Num_Motors):
                    self.left_hand_state_array[idx] = left_msg.angle_act[idx] / 1000.0  # millidegree → degree

            right_msg = self.right_hand_sub.Read()
            if right_msg is not None:
                for idx in range(Inspire_Num_Motors):
                    self.right_hand_state_array[idx] = right_msg.angle_act[idx] / 1000.0

            time.sleep(0.002)

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """
        Set current left, right hand motor target q
        """
        self.left_cmd.angle_set = (left_q_target * 1000.0).astype(np.int16).tolist()
        self.right_cmd.angle_set = (right_q_target * 1000.0).astype(np.int16).tolist()

        self.left_hand_pub.Write(self.left_cmd)
        self.right_hand_pub.Write(self.right_cmd)

    def control_process(self, left_hand_array, right_hand_array, left_hand_state_array, right_hand_state_array, dual_hand_data_lock=None, dual_hand_state_array=None, dual_hand_action_array=None):
        try:
            while self.running:
                start_time = time.time()

                # Read input hand tracking
                left_hand_mat = np.array(left_hand_array[:]).reshape(25, 3).copy()
                right_hand_mat = np.array(right_hand_array[:]).reshape(25, 3).copy()

                # Default q_target
                left_q_target = np.ones(Inspire_Num_Motors)
                right_q_target = np.ones(Inspire_Num_Motors)

                # Check if valid hand data
                if not np.all(right_hand_mat == 0.0) and not np.all(left_hand_mat[inspire_tip_indices[0]] == np.array([-1.13, 0.3, 0.15])):
                    ref_left_value = left_hand_mat[inspire_tip_indices]
                    ref_right_value = right_hand_mat[inspire_tip_indices]

                    left_q_target = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[self.hand_retargeting.right_dex_retargeting_to_hardware]
                    right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[self.hand_retargeting.right_dex_retargeting_to_hardware]

                    # Normalize to [0,1]
                    def normalize(val, min_val, max_val):
                        return np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0)

                    for idx in range(Inspire_Num_Motors):
                        if idx <= 3:
                            left_q_target[idx] = normalize(left_q_target[idx], 0.0, 1.7)
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 1.7)
                        elif idx == 4:
                            left_q_target[idx] = normalize(left_q_target[idx], 0.0, 0.5)
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 0.5)
                        elif idx == 5:
                            left_q_target[idx] = normalize(left_q_target[idx], -0.1, 1.3)
                            right_q_target[idx] = normalize(right_q_target[idx], -0.1, 1.3)

                # Update shared memory
                if dual_hand_state_array and dual_hand_action_array:
                    with dual_hand_data_lock:
                        state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))
                        action_data = np.concatenate((left_q_target, right_q_target))
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data

                # Publish control
                self.ctrl_dual_hand(left_q_target, right_q_target)

                elapsed = time.time() - start_time
                sleep_time = max(0, (1.0 / self.fps) - elapsed)
                time.sleep(sleep_time)
        finally:
            print("Inspire_Controller control_process ended.")
