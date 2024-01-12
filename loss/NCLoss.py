import torch
import torch.nn as nn

import numpy as np

class NCLoss(nn.Module):
    def __init__(self, tau=0.5, weight=None, **kwargs):
        super(NCLoss, self).__init__()
        self.tau = float(tau)
        self.weight = weight
        self.reweight = True if weight is not None else False
        print('Reweight: {}'.format(self.reweight))

        self.CE = nn.CrossEntropyLoss()

    def nc_loss(self, logits):
        pred = torch.softmax(logits, dim=1)
        batch_size = pred.size(0)
        if self.reweight:
            P_hat = torch.mm(pred * self.weight, pred.transpose(0, 1))
        else:
            P_hat = torch.mm(pred, pred.transpose(0, 1))
        loss = P_hat.norm(p='nuc') / batch_size
        return loss

    def forward(self, logits, targets, *args):
        ce = self.CE(logits, targets)
        nc = self.nc_loss(logits)
        loss = (1.0 - self.tau) * ce + self.tau * nc
        return loss, ce

if __name__ == '__main__':
    n_class = 10
    criterion = NCLoss(tau=0.5, device='cpu')


    x = torch.randn(4, n_class)
    y = torch.randint(0, 10, (4,))

    loss, ce = criterion(x, y)
    print(loss, ce)
