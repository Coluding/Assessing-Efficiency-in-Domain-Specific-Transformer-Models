import torch

# Parameters
sequence_length = 100
d_model = 64
window_size = 10  # Example window size


tensor = torch.rand(sequence_length, d_model)
pad_size = window_size - 1
left_pad = pad_size // 2
right_pad = pad_size - left_pad
d=3
# Pad the sequence
# We're padding along the sequence dimension (dim 0), hence (left_pad, right_pad)
padded_tensor = torch.nn.functional.pad(tensor, (0, 0, left_pad, right_pad))
# Create an index tensor to gather slices
window_inds = torch.arange(0, window_size * d, d)
idx = window_inds[None, :] + torch.arange(sequence_length - (window_size - 1) * d)[:, None]

print(idx.shape)
# Slice the tensor using advanced indexing
sliced_tensor = padded_tensor[idx]

# sliced_tensor now has the shape [91, 10, 64]
print(sliced_tensor.shape)