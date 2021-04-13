import torch
import torch.nn as nn
from Network.Loss import DiceBCELoss


class UncertaintyLoss(nn.Module):
    def __init__(self, mode):
        self.mode = mode
        super().__init__()

    def forward(self, x):
        """
        x: (batch, n, h, w, d) before sigmoid
        """
        var = torch.var(x, dim=1)  # (batch, h, w, d)
        if type(self.mode) is int:
            loss = var.mean()
            # loss = torch.nn.L1Loss()(var, torch.zeros(var.shape, device=var.device))

            loss *= self.mode
        elif self.mode == 'weight':
            mean = x.mean(dim=1)  # (batch, h, w, d)
            fg = (torch.sigmoid(mean).gt(0.5))
            loss = (var * fg).mean()
        else:
            raise ValueError
        return loss


class AllLoss(nn.Module):
    def __init__(self, mode=None, pos_weight=None):
        """
        mode: int or 'weight'
        pos_weight:
        """
        super().__init__()
        self.mode = mode
        self.uncertainty = UncertaintyLoss(mode)
        self.dicebce = DiceBCELoss(pos_weight)

    def forward(self, x, y):
        """
        x: (batch, 1, h, w, d) if uncertainty or (batch, n, h, w, d) before sigmoid
        y: (batch, 1, h, w, d) if uncertainty or (batch, n, h, w, d)
        """
        if self.mode is not None:
            loss, dice, bce = self.dicebce(x.mean(dim=1, keepdim=True), y)
            uncertainty = self.uncertainty(x)
            loss += uncertainty
            return loss, dice, bce, uncertainty
        else:
            loss, dice, bce = self.dicebce(x, y)
            return loss, dice, bce
