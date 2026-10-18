# rollout_buffer.py
import torch
from features_spec import GLOBAL_DIM, OBJECT_FEATURE_DIM, MAX_TOKENS

class RolloutBuffer:
    def __init__(self, capacity: int = 4096, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.num_obj = MAX_TOKENS - 1
        
        # Pre-allocate buffer tensors
        self.globals = torch.zeros((capacity, GLOBAL_DIM), dtype=torch.float32)
        self.obj_features = torch.zeros((capacity, self.num_obj, OBJECT_FEATURE_DIM), dtype=torch.float32)
        self.card_ids = torch.zeros((capacity, self.num_obj), dtype=torch.long)
        self.evo_ids = torch.zeros((capacity, self.num_obj), dtype=torch.long)
        self.token_types = torch.zeros((capacity, self.num_obj), dtype=torch.long)
        self.option_mask = torch.zeros((capacity, self.num_obj), dtype=torch.bool)
        self.valid_mask = torch.zeros((capacity, self.num_obj), dtype=torch.bool)
        self.pad_mask = torch.zeros((capacity, MAX_TOKENS), dtype=torch.bool)
        
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.log_probs = torch.zeros(capacity, dtype=torch.float32)
        self.values = torch.zeros(capacity, dtype=torch.float32)
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.dones = torch.zeros(capacity, dtype=torch.bool)
        
        self.advantages = torch.zeros(capacity, dtype=torch.float32)
        self.returns = torch.zeros(capacity, dtype=torch.float32)
        self.ptr = 0

    def add(self, parsed_state: dict, action: int, log_prob: float, value: float, reward: float, done: bool):
        if self.ptr >= self.capacity:
            return
        idx = self.ptr
        self.globals[idx] = torch.from_numpy(parsed_state["global_vec"])
        self.obj_features[idx] = torch.from_numpy(parsed_state["obj_features"])
        self.card_ids[idx] = torch.from_numpy(parsed_state["card_ids"])
        self.evo_ids[idx] = torch.from_numpy(parsed_state["evo_ids"])
        if "token_types" in parsed_state:
            self.token_types[idx] = torch.from_numpy(parsed_state["token_types"])
        else:
            self.token_types[idx] = torch.zeros(self.num_obj, dtype=torch.long)
            
        self.option_mask[idx] = torch.from_numpy(parsed_state["option_mask"])
        self.valid_mask[idx] = torch.from_numpy(parsed_state["valid_action_mask"])
        self.pad_mask[idx] = torch.from_numpy(parsed_state["padding_mask"])
        
        self.actions[idx] = action
        self.log_probs[idx] = log_prob
        self.values[idx] = value
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr >= self.capacity

    def get_batches(self, batch_size: int = 256):
        indices = torch.randperm(self.ptr)
        for start in range(0, self.ptr, batch_size):
            end = min(start + batch_size, self.ptr)
            b_idx = indices[start:end]
            if len(b_idx) < 16:
                continue
            yield {
                "global_vec": self.globals[b_idx].to(self.device),
                "obj_features": self.obj_features[b_idx].to(self.device),
                "card_ids": self.card_ids[b_idx].to(self.device),
                "evo_ids": self.evo_ids[b_idx].to(self.device),
                "token_types": self.token_types[b_idx].to(self.device),
                "option_mask": self.option_mask[b_idx].to(self.device),
                "valid_action_mask": self.valid_mask[b_idx].to(self.device),
                "padding_mask": self.pad_mask[b_idx].to(self.device),
                "actions": self.actions[b_idx].to(self.device),
                "old_log_probs": self.log_probs[b_idx].to(self.device),
                "advantages": self.advantages[b_idx].to(self.device),
                "returns": self.returns[b_idx].to(self.device),
            }

    def clear(self):
        self.ptr = 0