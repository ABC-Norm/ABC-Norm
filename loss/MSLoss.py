import torch
import torch.nn as nn

import numpy as np

class MSLoss(nn.Module):
    '''
    Minimize max singular value of y_pred.
    '''
    def __init__(self, tau=0.0, weight=None, **kwargs):
        super(MSLoss, self).__init__()
        self.tau = float(tau)
        self.weight = weight
        self.reweight = True if weight is not None else False
        print('Reweight: {}'.format(self.reweight))

        self.CE = nn.CrossEntropyLoss()

    def br_loss(self, logits):
        y_pred = torch.softmax(logits, dim=1)
        if self.reweight:
            y_pred = y_pred * self.weight
        loss = y_pred.norm(p=2)
        return loss

    def forward(self, logits, targets, *args):
        ce = self.CE(logits, targets)
        br = self.br_loss(logits)
        loss = (1.0 - self.tau) * ce + self.tau * br
        return loss, ce

if __name__ == '__main__':
    n_class = 10
    criterion = MSLoss(tau=0.5, device='cpu')


    x = torch.randn(4, n_class)
    y = torch.randint(0, 10, (4,))

    loss, ce = criterion(x, y)
    print(loss, ce)
