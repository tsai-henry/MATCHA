import time
import numpy as np
import threading
from multiprocessing import Process, Array
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from inspire_sdkpy import inspire_dds, inspire_hand_defaut
from teleop.robot_control.hand_retargeting import HandRetargeting, HandType

inspire_num_motors = 6
inspire_num_tactile_cells = 1062
inspire_tip_indices = [4, 9, 14, 19, 24]
command_topic_r = "rt/inspire_hand/ctrl/r"
command_topic_l = "rt/inspire_hand/ctrl/l"
state_topic_r = "rt/inspire_hand/state/r"
state_topic_l = "rt/inspire_hand/state/l"
touch_ropic_r = "rt/inspire_hand/touch/r"
touch_topic_l = "rt/inspire_hand/touch/l"


class ReplayInspireControllerRH56DFTP:
    def __init__(self):
        command_topic_l = "rt/inspire_hand/ctrl/l"
        command_topic_r = "rt/inspire_hand/ctrl/r"

        self.left_hand_pub = ChannelPublisher(command_topic_l, inspire_dds.inspire_hand_ctrl)
        self.right_hand_pub = ChannelPublisher(command_topic_r, inspire_dds.inspire_hand_ctrl)
        self.left_hand_pub.Init()
        self.right_hand_pub.Init()

        self.left_cmd = inspire_hand_defaut.get_inspire_hand_ctrl()
        self.right_cmd = inspire_hand_defaut.get_inspire_hand_ctrl()
        self.left_cmd.mode = 0b0001
        self.right_cmd.mode = 0b0001

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """
        Send joint position commands to both hands.
        """
        self.left_cmd.angle_set = (left_q_target * 1000.0).astype(np.int16).tolist()
        self.right_cmd.angle_set = (right_q_target * 1000.0).astype(np.int16).tolist()

        self.left_hand_pub.Write(self.left_cmd)
        self.right_hand_pub.Write(self.right_cmd)


class InspireControllerRH56DFTP:
    def __init__(self, left_hand_array, right_hand_array, dual_hand_data_lock=None, dual_hand_state_array=None, dual_hand_action_array=None, dual_hand_touch_array=None, fps=100.0, Unit_Test=False):
        print("Initializing InspireControllerRH56DFTP...")
        self.fps = fps
        self.Unit_Test = Unit_Test

        self.left_hand_pub = ChannelPublisher(command_topic_l, inspire_dds.inspire_hand_ctrl)
        self.right_hand_pub = ChannelPublisher(command_topic_r, inspire_dds.inspire_hand_ctrl)
        self.left_hand_pub.Init()
        self.right_hand_pub.Init()

        self.left_hand_state_sub = ChannelSubscriber(state_topic_l, inspire_dds.inspire_hand_state)
        self.right_hand_state_sub = ChannelSubscriber(state_topic_r, inspire_dds.inspire_hand_state)
        self.left_hand_state_sub.Init()
        self.right_hand_state_sub.Init()

        self.left_hand_touch_sub = ChannelSubscriber(touch_topic_l, inspire_dds.inspire_hand_touch)
        self.right_hand_touch_sub = ChannelSubscriber(touch_ropic_r, inspire_dds.inspire_hand_touch)
        self.left_hand_touch_sub.Init()
        self.right_hand_touch_sub.Init()

        self.left_hand_state_array = Array('d', inspire_num_motors, lock=True)
        self.right_hand_state_array = Array('d', inspire_num_motors, lock=True)
        self.left_hand_touch_array = Array('d', inspire_num_tactile_cells, lock=True)
        self.right_hand_touch_array = Array('d', inspire_num_tactile_cells, lock=True)

        self.shared_left_hand_array = left_hand_array
        self.shared_right_hand_array = right_hand_array
        self.dual_hand_data_lock = dual_hand_data_lock
        self.dual_hand_state_array = dual_hand_state_array
        self.dual_hand_action_array = dual_hand_action_array
        self.dual_hand_touch_array = dual_hand_touch_array

        if not self.Unit_Test:
            self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND)
        else:
            self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND_Unit_Test)

        self.left_cmd = inspire_hand_defaut.get_inspire_hand_ctrl()
        self.right_cmd = inspire_hand_defaut.get_inspire_hand_ctrl()
        self.left_cmd.mode = 0b0001
        self.right_cmd.mode = 0b0001

        self.running = True
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        self.subscribe_touch_thread = threading.Thread(target=self._subscribe_hand_touch)
        self.subscribe_touch_thread.daemon = True
        self.subscribe_touch_thread.start()

        while True:
            if any(self.right_hand_state_array) and any(self.left_hand_state_array):
                break
            print("[InspireControllerRH56DFTP] Waiting to subscribe DDS...")
            time.sleep(0.01)
        

        self.hand_control_process = Process(
            target=self.control_process,
            args=(self.shared_left_hand_array, self.shared_right_hand_array, self.left_hand_state_array, self.right_hand_state_array, self.left_hand_touch_array, self.right_hand_touch_array, self.dual_hand_data_lock, self.dual_hand_state_array, self.dual_hand_action_array, self.dual_hand_touch_array)
        )
        self.hand_control_process.daemon = True
        self.hand_control_process.start()

        print("InspireControllerRH56DFTP initialized OK.\n")

    def _subscribe_hand_state(self):
        while self.running:
            left_msg = self.left_hand_state_sub.Read()
            if left_msg is not None:
                for idx in range(inspire_num_motors):
                    self.left_hand_state_array[idx] = left_msg.angle_act[idx] / 1000.0  # millidegree → degree

            right_msg = self.right_hand_state_sub.Read()
            if right_msg is not None:
                for idx in range(inspire_num_motors):
                    self.right_hand_state_array[idx] = right_msg.angle_act[idx] / 1000.0

            time.sleep(0.002)
    
    def _subscribe_hand_touch(self):
        while self.running:
            left_msg = self.left_hand_touch_sub.Read()
            if left_msg is not None:
                self.left_hand_touch_array[:] = (np.array(self._flatten_touch_msg(left_msg)) / 4095.0).tolist() # normalize

            right_msg = self.right_hand_touch_sub.Read()
            if right_msg is not None:
                self.right_hand_touch_array[:] = (np.array(self._flatten_touch_msg(right_msg)) / 4095.0).tolist()

            time.sleep(0.002)
    
    @staticmethod
    def _flatten_touch_msg(touch_msg):
        # NOTE: palm_touch is column-major, bottom-up, left-to-right. 
        # all other touch data is row-major, left-to-right, top-to-bottom.
        return (
            touch_msg.fingerone_tip_touch +
            touch_msg.fingerone_top_touch +
            touch_msg.fingerone_palm_touch +
            touch_msg.fingertwo_tip_touch +
            touch_msg.fingertwo_top_touch +
            touch_msg.fingertwo_palm_touch +
            touch_msg.fingerthree_tip_touch +
            touch_msg.fingerthree_top_touch +
            touch_msg.fingerthree_palm_touch +
            touch_msg.fingerfour_tip_touch +
            touch_msg.fingerfour_top_touch +
            touch_msg.fingerfour_palm_touch +
            touch_msg.fingerfive_tip_touch +
            touch_msg.fingerfive_top_touch +
            touch_msg.fingerfive_middle_touch +
            touch_msg.fingerfive_palm_touch +
            touch_msg.palm_touch
        )

    @staticmethod
    def _unflatten_touch_arr(touch_arr):
        assert len(touch_arr) == inspire_num_tactile_cells, "Invalid touch array length"
        unflattened = {
            "fingerone_tip_touch": touch_arr[:9], 
            "fingerone_top_touch": touch_arr[9:105], 
            "fingerone_palm_touch": touch_arr[105:185], 
            "fingertwo_tip_touch": touch_arr[185:194], 
            "fingertwo_top_touch": touch_arr[194:290],
            "fingertwo_palm_touch": touch_arr[290:370], 
            "fingerthree_tip_touch": touch_arr[370:379], 
            "fingerthree_top_touch": touch_arr[379:475], 
            "fingerthree_palm_touch": touch_arr[475:555], 
            "fingerfour_tip_touch": touch_arr[555:564], 
            "fingerfour_top_touch": touch_arr[564:660], 
            "fingerfour_palm_touch": touch_arr[660:740], 
            "fingerfive_tip_touch": touch_arr[740:749], 
            "fingerfive_top_touch": touch_arr[749:845], 
            "fingerfive_middle_touch": touch_arr[845:854], 
            "fingerfive_palm_touch": touch_arr[854:950], 
            "palm_touch": touch_arr[950:1062],
        }
        return unflattened

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """
        Set current left, right hand motor target q
        """
        self.left_cmd.angle_set = (left_q_target * 1000.0).astype(np.int16).tolist()
        self.right_cmd.angle_set = (right_q_target * 1000.0).astype(np.int16).tolist()

        self.left_hand_pub.Write(self.left_cmd)
        self.right_hand_pub.Write(self.right_cmd)

    def control_process(self, left_hand_array, right_hand_array, left_hand_state_array, right_hand_state_array, left_hand_touch_array, right_hand_touch_array, dual_hand_data_lock=None, dual_hand_state_array=None, dual_hand_action_array=None, dual_hand_touch_array=None):
        try:
            while self.running:
                start_time = time.time()

                left_hand_mat = np.array(left_hand_array[:]).reshape(25, 3).copy()
                right_hand_mat = np.array(right_hand_array[:]).reshape(25, 3).copy()

                left_q_target = np.ones(inspire_num_motors)
                right_q_target = np.ones(inspire_num_motors)

                if not np.all(right_hand_mat == 0.0) and not np.all(left_hand_mat[inspire_tip_indices[0]] == np.array([-1.13, 0.3, 0.15])):
                    ref_left_value = left_hand_mat[inspire_tip_indices]
                    ref_right_value = right_hand_mat[inspire_tip_indices]

                    left_q_target = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[self.hand_retargeting.right_dex_retargeting_to_hardware]
                    right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[self.hand_retargeting.right_dex_retargeting_to_hardware]

                    # Normalize to [0,1]
                    def normalize(val, min_val, max_val):
                        return np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0)

                    for idx in range(inspire_num_motors):
                        if idx <= 3:
                            left_q_target[idx] = normalize(left_q_target[idx], 0.0, 1.7)
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 1.7)
                        elif idx == 4:
                            left_q_target[idx] = normalize(left_q_target[idx], 0.0, 0.5)
                            right_q_target[idx] = normalize(right_q_target[idx], 0.0, 0.5)
                        elif idx == 5:
                            left_q_target[idx] = normalize(left_q_target[idx], -0.1, 1.3)
                            right_q_target[idx] = normalize(right_q_target[idx], -0.1, 1.3)

                if dual_hand_state_array and dual_hand_action_array and dual_hand_touch_array:
                    with dual_hand_data_lock:
                        state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))
                        action_data = np.concatenate((left_q_target, right_q_target))
                        touch_data = np.concatenate((np.array(left_hand_touch_array[:]), np.array(right_hand_touch_array[:])))
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data
                        dual_hand_touch_array[:] = touch_data

                self.ctrl_dual_hand(left_q_target, right_q_target)

                elapsed = time.time() - start_time
                sleep_time = max(0, (1.0 / self.fps) - elapsed)
                time.sleep(sleep_time)
        finally:
            print("InspireControllerRH56DFTP control_process ended.")
