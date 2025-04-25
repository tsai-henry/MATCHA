import json 
import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

class EpisodeLoader:
    """
    Loads episode data from a JSON file written by teleop module.
    """
    def __init__(self, episode_path: str):
        self.ep_path = episode_path
        self.ep_idx = 0

        self.ep_data = json.load(open(self.ep_path, 'r'))['data']
        self.ep_len = len(self.ep_data)

    def __len__(self):
        return self.ep_len

    def __getitem__(self, idx: int):
        if not 0 <= idx < self.ep_len:
            raise IndexError(f"Index {idx} out of range for episode data of length {self.ep_len}")
        
        data_obj = self.ep_data[idx]

        return {
            "left_arm_state_qpos": data_obj["states"]["left_arm"]["qpos"],
            "left_arm_action_qpos": data_obj["actions"]["left_arm"]["qpos"],
            "right_arm_state_qpos": data_obj["states"]["right_arm"]["qpos"],
            "right_arm_action_qpos": data_obj["actions"]["right_arm"]["qpos"],

            "left_arm_state_qvel": data_obj["states"]["left_arm"]["qvel"],
            "left_arm_action_qvel": data_obj["actions"]["left_arm"]["qvel"],
            "right_arm_state_qvel": data_obj["states"]["right_arm"]["qvel"],
            "right_arm_action_qvel": data_obj["actions"]["right_arm"]["qvel"],
            
            "left_arm_state_torque": data_obj["states"]["left_arm"]["torque"],
            "left_arm_action_torque": data_obj["actions"]["left_arm"]["torque"],
            "right_arm_state_torque": data_obj["states"]["right_arm"]["torque"],
            "right_arm_action_torque": data_obj["actions"]["right_arm"]["torque"],
        }

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.ep_idx >= self.ep_len:
            raise StopIteration
        
        data_obj = self[self.ep_idx]
        self.ep_idx += 1

        return data_obj
