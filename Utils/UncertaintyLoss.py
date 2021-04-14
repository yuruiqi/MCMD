import torch
import torch.nn as nn
from Network.Loss import DiceBCELoss


class UncertaintyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        x: (batch, n, h, w, d) after sigmoid
        """
        var = torch.var(x, dim=1)  # (batch, h, w, d)
        loss = var.mean()
        # loss = torch.nn.L1Loss()(var, torch.zeros(var.shape, device=var.device))

        return loss


class AllLoss(nn.Module):
    def __init__(self, mode=None, weight=1):
        """
        mode: 'dicebce' or 'uncertainty'
        pos_weight:
        """
        super().__init__()
        self.mode = mode
        self.weight = weight
        self.uncertainty = UncertaintyLoss()
        self.dicebce = DiceBCELoss()

    def forward(self, x, y):
        """
        x: (batch, 1, h, w, d) or (batch, n, h, w, d) after sigmoid
        y: (batch, 1, h, w, d) or (batch, n, h, w, d)
        """
        if y.shape[1] < x.shape[1]:
            dim = len(y.shape[2:])
            y = y.repeat([1, x.shape[1]] + [1, ] * dim)

        loss, dice, bce = self.dicebce(x, y)
        if self.mode == 'dicebce':
            return loss, dice, bce
        elif self.mode == 'uncertainty':
            uncertainty = self.uncertainty(x) * self.weight
            loss += uncertainty
            return loss, dice, bce, uncertainty
