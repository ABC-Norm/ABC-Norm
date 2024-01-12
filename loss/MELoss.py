import torch
import torch.nn as nn

class MELoss(nn.Module):
    def __init__(self, tau=0.0, **kwargs):
        super(MELoss, self).__init__()
        self.tau = float(tau)
        self.CE = nn.CrossEntropyLoss()

    def me_loss(self, logits):
        p = torch.softmax(logits, dim=1)
        H = - (p * torch.log(p)).sum(dim=1)
        return H.mean()

    def forward(self, logits, targets, *args):
        ce = self.CE(logits, targets)
        me = self.me_loss(logits)
        loss = (1.0 - self.tau) * ce + self.tau * me
        return loss, ce

