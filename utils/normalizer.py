import torch
from torch import nn

class Normalizer(nn.Module):
    """Normalize fixed targets with non-trainable device-aware statistics."""

    def __init__(self, tensor, mean=None, std=None):
        super().__init__()
        if mean is None and std is None:
            self.register_buffer('mean', torch.mean(tensor, dim=0).detach())
            self.register_buffer('std', torch.std(tensor, dim=0).detach())
        else:
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean
