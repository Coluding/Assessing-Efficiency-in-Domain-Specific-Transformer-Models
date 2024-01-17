import einops
import torch
from typing import *
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Union, Set, Tuple, List
from transformers import TrainingArguments, set_seed
from dataclasses import dataclass


def my_set_seed(seed: int):
    """

    :param seed:
    """
    set_seed(seed=seed)
    torch.backends.cudnn.deterministic = True  # needed for reproducible experiments
    torch.backends.cudnn.benchmark = False

def training_args_from_yaml(yaml_file: str, key=None) -> TrainingArguments:
    """
    Helper function to load Huggingface tokenizers
    :param yaml_file:
    :return: tokenizer instance
    """
    import yaml
    with open(yaml_file, 'r') as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
            if key is not None:
                yaml_dict = yaml_dict[key]
        except yaml.YAMLError as exc:
            print(exc)

    return TrainingArguments(**yaml_dict)

@dataclass
class MyTrainingArguments(TrainingArguments):
    """
        Class to overload the default HF TrainingArguments in order to keep custom parameters in Tensorboard
    """
    pretrain_path: str = None
    freeze_weight: bool = None
    optimizer: str = None
    training_set_size: int = None
    layerwise_lr_decay_power: float = None

    training_set_random_seed: int = None
    valid_test_split_random_seed: int = None

    threshold_to_train_generator: float = None
    metric_to_train_generator: str = None

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """
        Sanitized serialization to use with TensorBoard’s hparams
        This method is strongly inspired by the HuggingFace's method
        See transformers\training_args.py
        """
        return to_sanitized_dict(self)


def to_sanitized_dict(self) -> Dict[str, Any]:
    """
    Sanitized serialization to use with TensorBoard’s hparams
    This method is strongly inspired by the HuggingFace's method
    See transformers\training_args.py
    """
    d = self.__dict__.copy()
    valid_types = [bool, int, float, str, torch.Tensor]
    return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}


def mem_use(title, fn, *args, **kwargs):
    # bookeeping
    import time

    start = time.time()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # actually run the function
    fn(*args, **kwargs)
    torch.cuda.synchronize()
    stop = time.time()

    # now report
    max_memory = torch.cuda.max_memory_allocated() // 2 ** 20
    print(f"{title} - Peak memory use: {max_memory}MB - {round((stop - start) * 1e6) / 1e3}ms")


def track_cuda_memory(title, function, *args, **kwargs):
    torch.cuda.synchronize()
    start_memory = torch.cuda.memory_allocated()
    result = function(*args, **kwargs)
    torch.cuda.synchronize()
    end_memory = torch.cuda.memory_allocated()
    memory_usage_bytes = end_memory - start_memory
    memory_usage_mb = memory_usage_bytes / (1024 ** 2)
    print(f"{title}: Memory Usage: {memory_usage_mb:.2f} MB")
    return result


def evaluate_cuda_memory(function, *args, **kwargs):
    """
    Evaluates the CUDA memory usage of a PyTorch function using torch.profiler.

    :param function: The function to be profiled.
    :param args: Arguments to be passed to the function.
    :param kwargs: Keyword arguments to be passed to the function.
    :return: A string containing the profiling results, specifically focusing on CUDA memory usage.
    """

    # Make sure CUDA is available
    if not torch.cuda.is_available():
        return "CUDA is not available. Cannot profile CUDA memory usage."

    # Enable CUDA memory profiling
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 profile_memory=True,
                 record_shapes=True) as prof:
        with record_function("model_inference"):
            function(*args, **kwargs)

    # Print the profiling results
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
    return prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10)


def local_1d_pattern_dilated(size: int, window_size: int, dilation_rate: int) -> torch.Tensor:
    """
    Generates a dilated 1D local attention pattern.

    This function creates a binary mask for local attention in 1D sequences, incorporating
    a dilation rate. This allows for flexible attention patterns, where each position can attend
    to positions within a dilated window around it. Refer to section 4.4 of the thesis for an explanation.

    :param size: The size of the sequence for which the mask is created.
    :param window_size: The base size of the attention window. This window size is modified
                        by the dilation rate.
    :param dilation_rate: The rate of dilation to apply, which spaces out the positions
                          that each token attends to within its window.
    :return: A square binary tensor of shape (size, size) representing the attention mask.
    """

    # Adjust window size for dilation and ensure it's odd for symmetry
    window_size = dilation_rate * (window_size - 1) + 1
    half_window = window_size // 2

    # Create a tensor representing position indices
    indices = torch.arange(size).unsqueeze(0)

    # Calculate the absolute difference between positions
    distance = torch.abs(indices - indices.T)

    # Create a base mask for the window
    # A position attends to other positions within half_window distance
    window_mask = (distance <= half_window)

    # Create a dilation mask
    # A position only attends to others at multiples of the dilation rate
    dilation_mask = ((distance % dilation_rate) == 0)

    # Combine both masks to get the final pattern
    # A position attends to another if it's within the dilated window
    mask = window_mask & dilation_mask

    return mask


def symmetric_global_token_mask(global_indices: torch.Tensor, size: int) -> torch.Tensor:
    """
    Creates a symmetric global token mask needed for global attention token. Refer to section 4.5 of the thesis for
    an explanation.

    :param global_indices: The indices of the query tensor for which global attention should be computed.
    :param size: Size of the sequence.
    :return: Boolean tensor of shape size x size.
    """
    global_mask = torch.zeros(size, size).bool()

    # make symmetric global mask
    global_mask[global_indices, :] = True
    global_mask[:, global_indices] = True

    return global_mask


# #################################################### deprecated ####################################################


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
    # TODO: slicing könnte memory intensive sein
    sliced_tensor = padded_tensor[:, full_window_indices]

    return sliced_tensor


def remove_global_attention_token_create_global_attention_tensor(embedding: torch.Tensor,
                                                                 global_attention_indices: torch.Tensor,
                                                                 receiver: torch.nn.Module = None) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Removes global attention tokens from an config and creates a global attention tensor.

    This function takes an config tensor and a sliced version of it, along with indices
    specifying global attention tokens. It creates a tensor containing only the global attention
    parts of the config and also returns a reduced version of the sliced config where the
    global attention tokens have been removed.

    :param embedding: The original config tensor.
    :param global_attention_indices:  Indices of tokens in the config that are considered
                                             as global attention tokens.
    :return: A tuple containing the global attention tensor and the reduced
                                       sliced config tensor
    """

    channels_to_keep = set(list(range(0, len(embedding[0])))).difference(set(global_attention_indices.tolist()))
    global_attention_tensor: torch.Tensor = embedding[:, global_attention_indices, ...]
    reduced_sliced_embedding: torch.Tensor = embedding[:, list(channels_to_keep), ...]

    if receiver is not None:
        receiver.local_token_indices = torch.tensor(list(channels_to_keep))

    return global_attention_tensor, reduced_sliced_embedding
