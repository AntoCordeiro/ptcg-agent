# test_smoke.py
import torch
from model import PTCGNet
from features_spec import GLOBAL_DIM, OBJECT_FEATURE_DIM, MAX_TOKENS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

batch_size = 4
num_obj_tokens = MAX_TOKENS - 1  # 140

# Generate dummy tensors
global_feat = torch.randn(batch_size, GLOBAL_DIM, device=device)
obj_feat = torch.randn(batch_size, num_obj_tokens, OBJECT_FEATURE_DIM, device=device)
card_ids = torch.randint(0, 1000, (batch_size, num_obj_tokens), device=device)
evo_ids = torch.randint(0, 1000, (batch_size, num_obj_tokens), device=device)

# Simulate 10 option tokens per sequence
option_mask = torch.zeros(batch_size, num_obj_tokens, dtype=torch.bool, device=device)
option_mask[:, :10] = True

valid_action_mask = torch.zeros(batch_size, num_obj_tokens, dtype=torch.bool, device=device)
valid_action_mask[:, :5] = True  # First 5 are legal choices

padding_mask = torch.zeros(batch_size, MAX_TOKENS, dtype=torch.bool, device=device)
padding_mask[:, 50:] = True  # Padded slots

model = PTCGNet().to(device)
probs, scores, values = model(
    global_feat, obj_feat, card_ids, evo_ids, 
    option_mask, valid_action_mask, padding_mask
)

print("Forward pass successful!")
print(f"Action probabilities shape: {probs.shape} (Sum: {probs.sum(dim=-1).detach().cpu().numpy()})")
print(f"Logits (scores) shape:      {scores.shape}")
print(f"Value predictions shape:    {values.shape}")