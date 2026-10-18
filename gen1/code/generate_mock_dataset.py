# generate_mock_dataset.py
import torch
from features_spec import GLOBAL_DIM, OBJECT_FEATURE_DIM, MAX_TOKENS

N_SAMPLES = 2000
num_obj_tokens = MAX_TOKENS - 1

print(f"Generating synthetic dataset with {N_SAMPLES} samples...")
global_vec = torch.randn(N_SAMPLES, GLOBAL_DIM)
obj_features = torch.randn(N_SAMPLES, num_obj_tokens, OBJECT_FEATURE_DIM)
card_ids = torch.randint(0, 1000, (N_SAMPLES, num_obj_tokens))
evo_ids = torch.randint(0, 1000, (N_SAMPLES, num_obj_tokens))

option_mask = torch.zeros(N_SAMPLES, num_obj_tokens, dtype=torch.bool)
option_mask[:, :12] = True

valid_action_mask = torch.zeros(N_SAMPLES, num_obj_tokens, dtype=torch.bool)
valid_action_mask[:, :6] = True

padding_mask = torch.zeros(N_SAMPLES, MAX_TOKENS, dtype=torch.bool)
padding_mask[:, 60:] = True

# Target index points to one of the first 6 valid options
target_indices = torch.randint(0, 6, (N_SAMPLES,), dtype=torch.long)

torch.save({
    "global_vec": global_vec,
    "obj_features": obj_features,
    "card_ids": card_ids,
    "evo_ids": evo_ids,
    "option_mask": option_mask,
    "valid_action_mask": valid_action_mask,
    "padding_mask": padding_mask,
    "target_indices": target_indices,
}, "parsed_replays.pt")

print("Saved dummy dataset to parsed_replays.pt.")