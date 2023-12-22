import torch

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