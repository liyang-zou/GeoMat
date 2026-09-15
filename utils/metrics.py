import torch
import torch.nn.functional as F


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def sce_loss(prediction, target, alpha=1, reduction='mean'):
    prediction = F.normalize(prediction)
    target = F.normalize(target)
    loss = (1 - (prediction * target).sum(dim=-1)).pow_(alpha)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")

def info_nce_loss(z1: torch.Tensor,
                           z2: torch.Tensor,
                           temperature: float = 0.5,
                           normalize: bool = True) -> torch.Tensor:
    """Symmetric cross-view InfoNCE loss for paired [B, C] embeddings."""
    assert z1.shape == z2.shape
    B = z1.size(0)

    if normalize:
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

    logits = (z1 @ z2.t()) / temperature
    targets = torch.arange(B, device=z1.device)

    loss_i = F.cross_entropy(logits, targets)
    loss_j = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_i + loss_j)

def accuracy_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return scalar accuracy from logits/one-hot labels or class indices."""
    if pred.dim() > 1:
        pred = torch.argmax(pred, dim=1)
    if target.dim() > 1:
        target = torch.argmax(target, dim=1)
    correct = (pred == target).float()
    return correct.mean()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
