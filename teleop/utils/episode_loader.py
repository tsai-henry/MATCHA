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

        return self.ep_data[idx]

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.ep_idx >= self.ep_len:
            raise StopIteration
        
        data_obj = self[self.ep_idx]
        self.ep_idx += 1

        return data_obj
