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

dofs = {
    "thumb": 4,
    "index": 3,
    "middle": 2,
    "ring": 1,
    "pinky": 0,
    "thumb_spread": 5
    }

# === Tactile processing and visualization ===
def get_patch_data(touch_array, section_name):
    """
    Extract sensor values from a specific tactile patch.
    
    Args:
        touch_array (dict): Output from _unflatten_touch_arr.
        section_name (str): Key name for the patch.

    Returns:
        np.ndarray: 1D array of normalized sensor values.
    """
    if section_name not in touch_array:
        raise ValueError(f"Section '{section_name}' not found in touch array.")
    return np.array(touch_array[section_name])


def compute_patch_averages(touch_array):
    """
    Compute the average value of each tactile patch.

    Args:
        touch_array (dict): Output from _unflatten_touch_arr.

    Returns:
        dict: Mapping from patch name to average pressure value.
    """
    return {k: float(np.mean(v)) for k, v in touch_array.items()}


def colorize_pressure(val, max_val=0.3):
    """
    Returns an ANSI color string depending on the intensity of the value relative to max_val.
    Blue = low, Red = high.
    
    Args:
        val (float): Pressure value.
        max_val (float): Maximum expected pressure for scaling.
    """
    normalized = np.clip(val / max_val, 0.0, 1.0)
    red = int(normalized * 255)
    blue = int((1.0 - normalized) * 255)
    return f"\033[38;2;{red};0;{blue}m{val:.4f}\033[0m"

def print_colored_patch_averages_both_hands(left_avg, right_avg):
    def colorize_pressure(p):
        intensity = int(p * 255)
        return f"\033[38;2;{intensity};0;0m{p:.3f}\033[0m"

    name_map = {
        "fingerone_top_touch": "Index top",
        "fingertwo_top_touch": "Middle top",
        "fingerthree_top_touch": "Ring top",
        "fingerfour_top_touch": "Pinky top",
        "fingerfive_top_touch": "Thumb top",
        "fingerone_tip_touch": "Index tip",
        "fingertwo_tip_touch": "Middle tip",
        "fingerthree_tip_touch": "Ring tip",
        "fingerfour_tip_touch": "Pinky tip",
        "fingerfive_tip_touch": "Thumb tip",
        "fingerfive_middle_touch": "Thumb mid",
        "fingerone_palm_touch": "Index pad",
        "fingertwo_palm_touch": "Middle pad",
        "fingerthree_palm_touch": "Ring pad",
        "fingerfour_palm_touch": "Pinky pad",
        "fingerfive_palm_touch": "Thumb pad",
        "palm_touch": "Palm"
    }

    all_keys = sorted(set(left_avg.keys()).union(right_avg.keys()))
    max_key_len = max(len(name_map.get(k, k)) for k in all_keys)

    print("\nAverage pressure per patch (Left vs Right):")
    for k in all_keys:
        name = name_map.get(k, k)
        left_val = left_avg.get(k, 0.0)
        right_val = right_avg.get(k, 0.0)
        left_col = colorize_pressure(np.clip(left_val, 0, 1))
        right_col = colorize_pressure(np.clip(right_val, 0, 1))
        print(f"{name.ljust(max_key_len)} : {left_col}   |   {right_col}")


def visualize_hand_touch_map(averaged_patches):
    """
    Visualize average tactile values for the whole hand.

    Args:
        averaged_patches (dict): Output from compute_patch_averages().
    """
    labels = list(averaged_patches.keys())
    values = list(averaged_patches.values())

    plt.figure(figsize=(10, 6))
    bars = plt.barh(labels, values, color='darkred')
    plt.xlabel("Average Normalized Pressure")
    plt.title("Tactile Sensor Averages per Patch")
    plt.grid(True, axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


def interpolate_three_fingers(source_finger, finger_positions):
    """
    Copies the source finger's position to the other non-thumb fingers.
    
    Args:
        source_finger (str): One of "index", "middle", "ring", or "pinky".
        finger_positions (list or np.ndarray): Array-like of 6 joint values [thumb, index, middle, ring, pinky, thumb_spread].

    Returns:
        np.ndarray: Modified copy of finger_positions with interpolation applied.
    """
    source_idx = dofs[source_finger]
    if source_finger == "thumb" or source_finger == "thumb_spread":
        raise ValueError("Cannot interpolate from thumb or thumb_spread")

    finger_positions = np.array(finger_positions).copy()
    for finger, idx in dofs.items():
        if finger not in ["thumb", "thumb_spread"] and idx != source_idx:
            finger_positions[idx] = finger_positions[source_idx]
    return finger_positions


class FollowerFingerController:
    def __init__(self, source_finger="index", step_size=0.02, epsilon=0.003, debug=False):
        self.source_finger = source_finger
        self.step_size = step_size
        self.epsilon = epsilon
        self.debug = debug
        self.follower_q = np.ones(6)  # Start fully extended
        self.patch_map = {
            "index": "fingerone_tip_touch",
            "middle": "fingertwo_tip_touch",
            "ring": "fingerthree_tip_touch",
            "pinky": "fingerfour_tip_touch",
            "thumb": "fingerfive_tip_touch"
        }

    def update(self, averaged_patches, robot_joint_state):
        """
        Updates follower fingers to match source finger pressure.

        Convention: 0.0 = fully curled, 1.0 = fully extended.

        Args:
            averaged_patches (dict): Patch name → avg pressure.
            robot_joint_state (np.ndarray): Current 6-DOF joint positions.

        Returns:
            np.ndarray: New 6-DOF joint targets.
        """
        robot_joint_state = np.array(robot_joint_state)
        output_q = robot_joint_state.copy()

        source_patch = self.patch_map[self.source_finger]
        source_pressure = averaged_patches.get(source_patch, 0.0)

        if self.debug:
            print(f"[DEBUG] Source finger pressure = {source_pressure:.4f}")

        for finger, idx in dofs.items():
            if finger in ["thumb_spread", self.source_finger]:
                continue

            patch = self.patch_map.get(finger)
            target_pressure = averaged_patches.get(patch, 0.0)
            pressure_diff = source_pressure - target_pressure

            if source_pressure < self.epsilon:
                self.follower_q[idx] = 1.0  # Fully extended
                if self.debug:
                    print(f"[DEBUG] {finger} set to fully extended (1.0)")
            elif pressure_diff > self.epsilon:
                # Not enough pressure → curl more (decrease value)
                self.follower_q[idx] = max(0.0, self.follower_q[idx] - self.step_size)
                if self.debug:
                    print(f"[DEBUG] Curling {finger} → {self.follower_q[idx]:.3f}")
            elif pressure_diff < -self.epsilon:
                # Too much pressure → straighten more (increase value)
                self.follower_q[idx] = min(1.0, self.follower_q[idx] + self.step_size)
                if self.debug:
                    print(f"[DEBUG] Straightening {finger} → {self.follower_q[idx]:.3f}")
            else:
                if self.debug:
                    print(f"[DEBUG] {finger} pressure matched (within epsilon)")

            output_q[idx] = self.follower_q[idx]

        if self.debug:
            print(f"[DEBUG] Output joint command: {output_q}")

        return output_q




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

                    left_q_target = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[self.hand_retargeting.left_dex_retargeting_to_hardware]
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

                # Interpolate fingers when testing disability

                # Index finger position cloning
                # source_finger = "index"
                # left_q_target = interpolate_three_fingers(source_finger, left_q_target)
                # right_q_target = interpolate_three_fingers(source_finger, right_q_target)

                # Assume you have already collected this from shared memory
                # Index finger pressure cloning
                # Split dual hand array
                left_hand_touch = dual_hand_touch_array[:1062]
                right_hand_touch = dual_hand_touch_array[1062:]

                # Unpack and parse touch arrays
                left_touch_dict = self._unflatten_touch_arr(left_hand_touch)
                right_touch_dict = self._unflatten_touch_arr(right_hand_touch)

                # 1. Get sensor values from the "fingerone_tip_touch" patch
                left_tip_data = get_patch_data(left_touch_dict, "fingerone_tip_touch")
                right_tip_data = get_patch_data(right_touch_dict, "fingerone_tip_touch")

                # 2. Compute average patch values for each hand
                left_avg_patches = compute_patch_averages(left_touch_dict)
                right_avg_patches = compute_patch_averages(right_touch_dict)

                # 3. Print side-by-side comparison
                print_interval = 100  # print every 30 frames (~3/sec if fps=100)
                frame_count = 0

                # Inside your control_process loop:
                frame_count += 1
                if frame_count % print_interval == 0:
                    print("Left hand  - Sensor values in 'fingerone_tip_touch':", left_tip_data)
                    print("Right hand - Sensor values in 'fingerone_tip_touch':", right_tip_data)

                    print_colored_patch_averages_both_hands(left_avg_patches, right_avg_patches)

                    print("[DEBUG] Adjusting fingers...")

                # # 3. Visualize the patch averages across the hand
                # visualize_hand_touch_map(averaged_patches)
                # 4. Interpolate finger positions based on pressure
                # left_q_target = adjust_fingers_by_pressure(left_avg_patches, "index", dual_hand_state_array, left_q_target)
                right_finger_positions = np.array(dual_hand_state_array[6:])
                follower_controller = FollowerFingerController(source_finger="index", debug=True)

                # Inside control_process:
                right_q_target = follower_controller.update(right_avg_patches, right_hand_state_array[:6])
                self.ctrl_dual_hand(left_q_target, right_q_target)

                elapsed = time.time() - start_time
                sleep_time = max(0, (1.0 / self.fps) - elapsed)
                time.sleep(sleep_time)
        finally:
            print("InspireControllerRH56DFTP control_process ended.")
