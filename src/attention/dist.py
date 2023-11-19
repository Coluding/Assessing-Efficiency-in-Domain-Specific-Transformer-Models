import torch

def create_dilated_window_attention_mask(size, window_size, dilation_rate):
    # Ensure window size is odd for symmetry
    window_size = dilation_rate * window_size
    window_size = window_size + 1 if window_size % 2 == 0 else window_size
    half_window = window_size // 2


    # Create a tensor representing position indices
    indices = torch.arange(size).unsqueeze(0)

    # Calculate the absolute difference between positions
    distance = torch.abs(indices - indices.T)

    # Create a base mask for the window
    window_mask = (distance < half_window)

    # Create a dilation mask
    dilation_mask = ((distance % dilation_rate) == 0)

    # Combine both masks
    mask = window_mask & dilation_mask

    return mask

# Usage
size = 2000
window_size = 5  # Example window size
dilation_rate = 2
mask = create_dilated_window_attention_mask(size, window_size, dilation_rate)

print(mask[2, :20])
