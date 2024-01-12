import torch
import torch.nn as nn
import numpy as np

from .LALoss import LALoss

class LABRLoss(nn.Module):
    def __init__(self, tau=0.0, weight=None, **kwargs):
        super(LABRLoss, self).__init__()
        self.tau = float(tau)
        self.CE = LALoss(weight=weight)

    def br_loss(self, logits):
        y_pred = torch.softmax(logits, dim=1)
        loss = y_pred.pow(2).sum(dim=1).mean()
        return loss

    def forward(self, logits, targets, *args):
        ce, _ = self.CE(logits, targets)
        br = self.br_loss(logits)
        loss = (1.0 - self.tau) * ce + self.tau * br
        return loss, ce

if __name__ == '__main__':
    n_class = 10
    criterion = BRLoss(tau=0.5, device='cpu')


    x = torch.randn(4, n_class)
    y = torch.randint(0, 10, (4,))

    loss, ce = criterion(x, y)
    print(loss, ce)
