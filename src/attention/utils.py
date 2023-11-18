import einops
import torch
from einops import rearrange, repeat
from typing import Union, Set, Tuple, List


def slice_tensor_in_windows(tensor: torch.Tensor, window_size: int, dilation_rate: int
                            ):
    """
    Slices a given tensor into windows of a specified size with a given dilation rate.

    This function pads the input tensor on both sides and then slices it into windows.
    Each token in the sequence will have a window of tokens around it, considering the specified window size and dilation rate.
    The padding is done such that the start and end tokens also have full windows.

    :param tensor: The input tensor to be sliced. Shape: (batch_size, sequence_length, d_model)
    :param window_size: The size of each window.
    :param dilation_rate: The step size between elements in each window.
    :return:  A tensor of shape (batch_size, sequence_length, window_size, d_model), where each token is associated with a window of tokens around it.

    Note:   The function currently pads the start of the sequence with zeros, meaning that the start token
        will attend to zeros. An alternative approach could be using a fixed window for the start tokens.
    """
    batch_size, sequence_length, d_model = tensor.shape

    # Calculate the total padding size
    pad_size = (window_size - 1) * dilation_rate
    left_pad = pad_size // 2
    right_pad = pad_size - left_pad
    # Pad the tensor
    padded_tensor = torch.nn.functional.pad(tensor, (0, 0, left_pad, right_pad))

    # create the 2d window indices with dilation rate as step size
    window_indices = torch.arange(0, (window_size) * dilation_rate, dilation_rate)

    # create 2d matrix of shifted window indices
    full_window_indices = window_indices[None, :] + torch.arange(sequence_length)[:, None]

    # use indices to create a tensor of shape sequence_length x window_size x d_model each token has a different
    # window currently I am padding such that the start token will attend to zeros, but maybe a fixed window for the
    # start tokens would be also an option
    sliced_tensor = padded_tensor[:, full_window_indices]

    return sliced_tensor


def remove_global_attention_token_create_global_attention_tensor(embedding: torch.Tensor,
                                                                 sliced_embedding: torch.Tensor,
                                                                 global_attention_indices = torch.Tensor) \
        -> Tuple[torch.Tensor]:
    """
    Removes global attention tokens from an embedding and creates a global attention tensor.

    This function takes an embedding tensor and a sliced version of it, along with indices
    specifying global attention tokens. It creates a tensor containing only the global attention
    parts of the embedding and also returns a reduced version of the sliced embedding where the
    global attention tokens have been removed.

    :param embedding: The original embedding tensor.
    :param sliced_embedding:  A sliced version of the original embedding tensor.
    :param global_attention_indices:  Indices of tokens in the embedding that are considered
                                             as global attention tokens.
    :return: A tuple containing the global attention tensor and the reduced
                                       sliced embedding tensor
    """

    channels_to_keep = set(list(range(0, len(embedding[0])))).difference(set(global_attention_indices.tolist()))
    global_attention_tensor = embedding[:, global_attention_indices, ...]
    reduced_sliced_embedding = sliced_embedding[:, list(channels_to_keep), ...]

    return global_attention_tensor, reduced_sliced_embedding






