import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F



def plot_softmax_bar(tensor):
    """
    Plots a bar chart of the softmax values of a PyTorch tensor.

    Args:
    tensor (torch.Tensor): The input tensor.
    """
    # Apply softmax to the tensor
    softmax_values = F.softmax(tensor, dim=-1).flatten()

    # Plot bar chart
    plt.bar(range(len(softmax_values)), softmax_values.detach().numpy())
    plt.title('Bar Chart of Softmax Values')
    plt.xlabel('Index')
    plt.ylabel('Softmax Value')
    plt.show()

def plot_softmax_histogram(tensor):
    """
    Plots a histogram of the softmax values of a PyTorch tensor.

    Args:
    tensor (torch.Tensor): The input tensor.
    """
    # Apply softmax to the tensor
    softmax_values = F.softmax(tensor, dim=-1).flatten()

    # Plot histogram
    plt.hist(softmax_values.detach().numpy(), bins=50, facecolor='blue', alpha=0.7)
    plt.title('Histogram of Softmax Values')
    plt.xlabel('Softmax Value')
    plt.ylabel('Frequency')
    plt.show()


def get_default_device():
    """
    Pick GPU if its available
    :return: None
    """
    if torch.cuda.is_available:
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    else:
        return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """
     Wrap a dataloader to move data to device
     """
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """
        Yield a batch of data after moving it to the device

        :return: Data transferred to specific device
        """
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """
        Number of batches

        """
        return len(self.dl)